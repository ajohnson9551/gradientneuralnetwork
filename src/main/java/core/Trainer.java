package core;

import java.util.Random;

public class Trainer {

    private final double trainingRate;
    private final Network net;
    private final Fitness trainFit;
    private final double percentToDo;
    private final boolean stochastic;
    private final int stochasticBatchSize;

    private final int ram;
    private final double[] mses;

    private final Random rand = new Random();
    private final Utility util = Utility.getUtility();

    public Trainer(double trainingRate, Network net, Fitness trainFit, double percentToDo, boolean stochastic, int stochasticBatchSize) {
        this.trainingRate = trainingRate;
        this.net = net;
        this.trainFit = trainFit;
        this.percentToDo = percentToDo;
        this.stochastic = stochastic;
        this.stochasticBatchSize = stochasticBatchSize;
        this.ram = 0;
        mses = null;
    }

    public Trainer(double trainingRate, Network net, Fitness trainFit, double percentToDo, boolean stochastic, int stochasticBatchSize, int ram) {
        this.trainingRate = trainingRate;
        this.net = net;
        this.trainFit = trainFit;
        this.percentToDo = percentToDo;
        this.stochastic = stochastic;
        this.stochasticBatchSize = stochasticBatchSize;
        this.ram = ram;
        mses = new double[ram];
    }

    public double[][] train(int cycles) {
        double[][] data = new double[2][cycles];
        for (int n = 0; n < cycles; n++) {
            if (!stochastic) {
                data[0][n] = trainFit.mse(net);
                data[1][n] = trainFit.percentCorrect(net);
            }
            if (net instanceof BasicNetwork) {
                trainStep((BasicNetwork) net);
            } else if (net instanceof LayeredNetwork) {
                trainStep((LayeredNetwork) net);
            }
            String cycleText = "Cycle " + n + "/" + cycles;
            if (!stochastic) {
                cycleText += " completed, mse = " + util.roundString(data[0][n])
                        + ", percent correct = " + util.roundString(100.0 * data[1][n]);
            }
            System.out.println(cycleText);
        }
        return data;
    }

    public void trainStep(LayeredNetwork net) {
        int numLayers = net.getNumHiddenLayers() + 2;

        double[][][] gradAs = net.getEmptyGradAs();
        double[][] gradBs = net.getEmptyGradBs();

        double[][][] As = net.getWeights();
        double[][] bs = net.getOffsets();

        int[] ns = new int[numLayers]; // number of nodes at each layer
        ns[0] = As[0][0].length; // number of inputs
        for (int i = 1; i < ns.length; i++) {
            ns[i] = As[i - 1].length;
        }

        int numOutputs = ns[numLayers - 1];

        double[] d = new double[numOutputs];

        double[][] data = trainFit.getData();
        double[][] answers = trainFit.getAnswers();
        int dataLength = (int) (data.length * percentToDo);
        int toDo = stochastic ? stochasticBatchSize : dataLength;
        int startAt = stochastic ? rand.nextInt(dataLength) : 0;

        double scaling = toDo * numOutputs;
        scaling = 2.0 / scaling;

        for (int k = startAt; k < startAt + toDo; k++) {
            double[][][][][] Fa = new double[numLayers - 1][][][][];
            double[][][][] Fb = new double[numLayers - 1][][][];

            double[][] alpha = new double[numLayers - 1][];
            double[][] y = new double[numLayers][];

            for (int l = 0; l < numLayers - 1; l++) {
                alpha[l] = new double[ns[l + 1]];
                Fa[l] = new double[ns[l + 1]][l + 1][][];
                Fb[l] = new double[ns[l + 1]][l + 1][];
                for (int i = 0; i < ns[l + 1]; i++) {
                    for (int m = 0; m <= l; m++) {
                        Fa[l][i][m] = new double[ns[m + 1]][ns[m]];
                        Fb[l][i][m] = new double[ns[m + 1]];
                    }
                }
            }
            double[] x = data[k % data.length];
            for (int l = 0; l < numLayers; l++) {
                y[l] = net.evaluate(x, l);
                if (l < numLayers - 1) {
                    for (int i = 0; i < ns[l + 1]; i++) {
                        alpha[l][i] = util.fastSigmoidPrime(util.dotProd(As[l][i], y[l]) + bs[l][i]);
                    }
                }
            }
            for (int i = 0; i < numOutputs; i++) {
                d[i] = y[numLayers - 1][i] - answers[k % data.length][i];
            }
            for (int l = 0; l < numLayers - 1; l++) {
                // compute Fa and Fb one layer at a time from the bottom up
                for (int i = 0; i < ns[l + 1]; i++) {
                    // compute each output of Fa and Fb
                    for (int m = 0; m < l; m++) {
                        // Fa and Fb contain derivatives for a and b from all layers below (for l > 0)
                        for (int r = 0; r < ns[m + 1]; r++) {
                            // chain rule occurs here for a with optimization at l - 1
                            if (m == l - 1) {
                                for (int j = 0; j < ns[m]; j++) {
                                    Fa[l][i][m][r][j] = alpha[l][i] * As[l][i][r] * Fa[l - 1][r][m][r][j];
                                }
                            } else {
                                for (int j = 0; j < ns[m]; j++) {
                                    for (int j1 = 0; j1 < ns[l]; j1++) {
                                        Fa[l][i][m][r][j] += As[l][i][j1] * Fa[l - 1][j1][m][r][j];
                                    }
                                    Fa[l][i][m][r][j] *= alpha[l][i];
                                }
                            }
                            // chain rule occurs here for b with optimization at l - 1
                            if (m == l - 1) {
                                Fb[l][i][m][r] = alpha[l][i] * As[l][i][r] * Fb[l - 1][r][m][r];
                            } else {
                                for (int j1 = 0; j1 < ns[l]; j1++) {
                                    Fb[l][i][m][r] += As[l][i][j1] * Fb[l - 1][j1][m][r];
                                }
                                Fb[l][i][m][r] *= alpha[l][i];
                            }
                        }
                    }
                    // Fa and Fb also contain derivatives for a and b at same layer
                    for (int j = 0; j < ns[l]; j++) {
                        Fa[l][i][l][i][j] = alpha[l][i] * y[l][j];
                    }
                    Fb[l][i][l][i] = alpha[l][i];
                }
            }
            // sum all outputs for gradients
            for (int l = 0; l < numLayers - 2; l++) {
                for (int i = 0; i < numOutputs; i++) {
                    for (int r = 0; r < ns[l + 1]; r++) {
                        for (int j = 0; j < ns[l]; j++) {
                            gradAs[l][r][j] += d[i] * Fa[numLayers - 2][i][l][r][j];
                        }
                        gradBs[l][r] += d[i] * Fb[numLayers - 2][i][l][r];
                    }
                }
            }
            // small optimization when l = numLayers - 2
            for (int i = 0; i < numOutputs; i++) {
                for (int j = 0; j < ns[numLayers - 2]; j++) {
                    gradAs[numLayers - 2][i][j] += d[i] * Fa[numLayers - 2][i][numLayers - 2][i][j];
                }
                gradBs[numLayers - 2][i] += d[i] * Fb[numLayers - 2][i][numLayers - 2][i];
            }
        }
        net.gradChange(gradAs, gradBs, scaling * trainingRate);
    }

    @Deprecated
    public void trainStep1(LayeredNetwork net) {
        // currently just for a single hidden layer
        double[][][] gradAs = net.getEmptyGradAs();
        double[][] gradBs = net.getEmptyGradBs();

        double[][][] As = net.getWeights();
        double[][] bs = net.getOffsets();
        int n = As[As.length - 1].length; // number of outputs
        int m = As[0].length; // nodes in first (only) hidden layer
        int u = As[0][0].length; // number of inputs

        double[][] data = trainFit.getData();
        double[][] answers = trainFit.getAnswers();
        int dataLength = (int) (data.length * percentToDo);
        int toDo = stochastic ? stochasticBatchSize : dataLength;
        int startAt = stochastic ? rand.nextInt(dataLength) : 0;

        double[] c = new double[m];
        double[] q = new double[m];
        double[] e = new double[m];
        double[] alpha;
        double[] r = new double[n];
        double[] d = new double[n];
        double[] w = new double[n];
        double[][] z = new double[n][m];

        double scaling = toDo * n;
        scaling = 2.0 / scaling;

        for (int k = startAt; k < startAt + toDo; k++) {
            double[] x = data[k % data.length];
            alpha = net.evaluate(x, 1);
            for (int i = 0; i < n; i++) {
                r[i] = util.dotProd(As[1][i], alpha) + bs[1][i];
                d[i] = util.fastSigmoid(r[i]) - answers[k % data.length][i];
                w[i] = util.fastSigmoidPrime(r[i]) * d[i];
            }
            for (int j = 0; j < m; j++) {
                c[j] = util.dotProd(As[0][j], x) + bs[0][j];
                q[j] = util.fastSigmoid(c[j]);
                e[j] = util.fastSigmoidPrime(c[j]);
                for (int i = 0; i < n; i++) {
                    z[i][j] = As[1][i][j] * e[j] * w[i];
                }

            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    for (int k1 = 0; k1 < u; k1++) {
                        gradAs[0][j][k1] += z[i][j] * x[k1];
                    }
                    gradAs[1][i][j] += w[i] * q[j];
                    gradBs[0][j] += z[i][j];
                }
                gradBs[1][i] += w[i];
            }
        }
        net.gradChange(gradAs, gradBs, scaling * trainingRate);
    }

    public void trainStep(BasicNetwork net) {
        double[][] gradA = net.getEmptyGradA();
        double[] gradB = net.getEmptyGradB();

        double[][] A = net.getWeights();
        double[] b = net.getOffset();
        int n = A.length;

        double[][] data = trainFit.getData();
        double[][] answers = trainFit.getAnswers();
        int toDo = (int) (data.length * percentToDo);

        double[] c = new double[n];
        double[] d = new double[n];
        double[] e = new double[n];
        for (int k = 0; k < toDo; k++) {
            if (stochastic) {
                k = rand.nextInt(toDo);
            }
            double[] x = data[k];
            // compute c, d, and e
            for (int i = 0; i < n; i++) {
                c[i] = util.dotProd(A[i], x) + b[i];
                d[i] = util.fastSigmoid(c[i]) - answers[k][i];
                e[i] = util.fastSigmoidPrime(c[i]);
            }
            // compute gradA at x
            for (int i = 0; i < gradA.length; i++) {
                for (int j = 0; j < gradA[0].length; j++) {
                    gradA[i][j] += e[i] * x[j] * d[i];
                }
            }
            // compute gradB at x
            for (int i = 0; i < gradB.length; i++) {
                gradB[i] += e[i] * d[i];
            }
            if (stochastic) {
                break;
            }
        }
        // constants
        double scaling = stochastic ? (double) n : (double) (toDo * n);
        for (int i = 0; i < gradA.length; i++) {
            for (int j = 0; j < gradA[0].length; j++) {
                gradA[i][j] = 2.0 * gradA[i][j] / scaling;
            }
            gradB[i] = 2.0 * gradB[i] / scaling;
        }
        net.gradChange(gradA, gradB, trainingRate);
    }
}
