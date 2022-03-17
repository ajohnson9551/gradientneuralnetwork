package core;

import java.util.Random;

public class Trainer {

    private double trainingRate;
    private final Network net;
    private final Fitness trainFit;
    private final double percentToDo;
    private final boolean stochastic;
    private final int stochasticBatchSize;

    private final int ram;
    private final double[] mses;
    private double alltimeMse = 0;
    private int mseIndex = 0;
    private int cycle = 0;

    private final boolean expAcc;

    private final double momentum;

    private final Random rand = new Random();
    private final Utility util = Utility.getUtility();

    // assumes using a layered network
    private double[][][] lastAs;
    private double[][] lastBs;

    public Trainer(double trainingRate, Network net, Fitness trainFit, double percentToDo, boolean stochastic, int stochasticBatchSize) {
        this.trainingRate = trainingRate;
        this.net = net;
        this.trainFit = trainFit;
        this.percentToDo = percentToDo;
        this.stochastic = stochastic;
        this.stochasticBatchSize = stochasticBatchSize;
        this.ram = 0;
        mses = null;
        expAcc = false;
        momentum = 0.0;
    }

    public Trainer(double trainingRate, Network net, Fitness trainFit, double percentToDo, boolean stochastic, int stochasticBatchSize, int ram, boolean expAcc, double momentum) {
        this.trainingRate = trainingRate;
        this.net = net;
        this.trainFit = trainFit;
        this.percentToDo = percentToDo;
        this.stochastic = stochastic;
        this.stochasticBatchSize = stochasticBatchSize;
        this.ram = ram;
        mses = new double[ram];
        this.expAcc = expAcc;
        this.momentum = momentum;
    }

    public double[][] train(int cycles) {
        double[][] data = new double[2][cycles];
        if (momentum > 0 && net instanceof LayeredNetwork) {
            lastAs = ((LayeredNetwork) net).getEmptyGradAs();
            lastBs = ((LayeredNetwork) net).getEmptyGradBs();
        }
        while (cycle < cycles) {
            if (!stochastic) {
                data[0][cycle] = trainFit.mse(net);
                data[1][cycle] = trainFit.percentCorrect(net);
            }
            if (net instanceof BasicNetwork) {
                trainStep((BasicNetwork) net);
            } else if (net instanceof LayeredNetwork) {
                int numHiddenLayers = ((LayeredNetwork) net).getNumHiddenLayers();
                if (numHiddenLayers == 2) {
                    trainStep2((LayeredNetwork) net);
                } else {
                    trainStep((LayeredNetwork) net);
                }
            }
            String cycleText = "Cycle " + cycle + "/" + cycles;
            if (!stochastic) {
                cycleText += " completed, mse = " + util.roundString(data[0][cycle])
                        + ", percent correct = " + util.roundString(100.0 * data[1][cycle]);
            }
            if (ram > 0) {
                cycleText += ", Run Avg Mse = " + util.avgString(mses);
                cycleText += ", All Avg Mse = " + util.roundString(alltimeMse);
            }
            System.out.println(cycleText);
            cycle++;
        }
        return data;
    }

    public void trainStep2(LayeredNetwork net) {
        int numLayers = net.getNumHiddenLayers() + 2;
        assert numLayers == 4;

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

        double scaling = toDo * numOutputs;
        scaling = 2.0 / scaling;

        double[] x = null;
        double[] ans = null;

        for (int k = 0; k < toDo; k++) {
            double[][] alpha = new double[numLayers - 1][];
            double[][] beta = new double[numOutputs][ns[1]];
            double[][] y = new double[numLayers][];

            for (int l = 0; l < numLayers - 1; l++) {
                alpha[l] = new double[ns[l + 1]];
            }
            int k1 = rand.nextInt(dataLength);
            x = data[k1];
            ans = answers[k1];
            for (int l = 0; l < numLayers; l++) {
                y[l] = net.evaluate(x, l);
                if (l < numLayers - 1) {
                    for (int i = 0; i < ns[l + 1]; i++) {
                        alpha[l][i] = util.fastSigmoidPrime(util.dotProd(As[l][i], y[l]) + bs[l][i]);
                    }
                }
            }
            for (int i = 0; i < numOutputs; i++) {
                for (int r = 0; r < ns[1]; r++) {
                    for (int j1 = 0; j1 < ns[2]; j1++) {
                        beta[i][r] += As[2][i][j1] * alpha[1][j1] * As[1][j1][r];
                    }
                    beta[i][r] *= alpha[0][r];
                }
            }
            for (int i = 0; i < numOutputs; i++) {
                d[i] = alpha[2][i] * (y[numLayers - 1][i] - ans[i]);
            }
            // sum all outputs for gradients
            // l = 0
            double q;
            for (int i = 0; i < numOutputs; i++) {
                for (int r = 0; r < ns[1]; r++) {
                    q = d[i] * beta[i][r];
                    gradBs[0][r] += q;
                    for (int j = 0; j < ns[0]; j++) {
                        gradAs[0][r][j] += q * y[0][j];
                    }
                }
            }
            // l = 1
            double p;
            for (int i = 0; i < numOutputs; i++) {
                for (int r = 0; r < ns[2]; r++) {
                    p = d[i] * As[2][i][r] * alpha[1][r];
                    gradBs[1][r] += p;
                    for (int j = 0; j < ns[1]; j++) {
                        gradAs[1][r][j] += p * y[1][j];
                    }
                }
            }
            //l = 2
            for (int i = 0; i < numOutputs; i++) {
                gradBs[2][i] += d[i];
                for (int j = 0; j < ns[2]; j++) {
                    gradAs[2][i][j] += d[i] * y[1][j];
                }
            }
        }
        double mse = 0.0;
        double rateMod = 1;
        if (ram > 0 && mses != null) {
            mse = util.mse(net.evaluate(x), ans);
            mses[mseIndex] = mse;
            mseIndex = (mseIndex + 1) % ram;
            alltimeMse *= ((double) cycle + 1.0) / ((double) cycle + 2.0);
            alltimeMse += mse / ((double) cycle + 2.0);
            rateMod = expAcc && stochasticBatchSize == 1 ? mse / 0.01 : 1.0;
            if (rateMod < 0.1) {
                rateMod = 0.1;
            } else if (rateMod > 2) {
                rateMod = 2;
            }
        }
        if (momentum > 0) {
            for (int l = 0; l < numLayers - 2; l++) {
                for (int r = 0; r < ns[l + 1]; r++) {
                    for (int j = 0; j < ns[l]; j++) {
                        gradAs[l][r][j] += momentum * lastAs[l][r][j];
                    }
                    gradBs[l][r] += momentum * lastBs[l][r];
                }
            }
            lastAs = gradAs;
            lastBs = gradBs;
        }
        net.gradChange(gradAs, gradBs, scaling * rateMod * trainingRate);
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

        double[] x = null;
        double[] ans = null;

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
            x = data[k % data.length];
            ans = answers[k % data.length];
            for (int l = 0; l < numLayers; l++) {
                y[l] = net.evaluate(x, l);
                if (l < numLayers - 1) {
                    for (int i = 0; i < ns[l + 1]; i++) {
                        alpha[l][i] = util.fastSigmoidPrime(util.dotProd(As[l][i], y[l]) + bs[l][i]);
                    }
                }
            }
            for (int i = 0; i < numOutputs; i++) {
                d[i] = y[numLayers - 1][i] - ans[i];
            }
            double sum;
            for (int l = 0; l < numLayers - 1; l++) {
                // compute Fa and Fb one layer at a time from the bottom up
                for (int i = 0; i < ns[l + 1]; i++) {
                    // compute each output of Fa and Fb
                    for (int m = 0; m < l - 1; m++) {
                        // Fa and Fb contain derivatives for a and b from all layers below (for l > 0)
                        for (int r = 0; r < ns[m + 1]; r++) {
                            // chain rule occurs here for a with optimization at l - 1
                            if (m == l - 1) {
                                for (int j = 0; j < ns[m]; j++) {
                                    Fa[l][i][m][r][j] = alpha[l][i] * As[l][i][r] * Fa[l - 1][r][m][r][j];
                                }
                            } else {
                                for (int j = 0; j < ns[m]; j++) {
                                    sum = 0;
                                    for (int j1 = 0; j1 < ns[l]; j1++) {
                                        sum += As[l][i][j1] * Fa[l - 1][j1][m][r][j];
                                    }
                                    Fa[l][i][m][r][j] = sum * alpha[l][i];
                                }
                            }
                            // chain rule occurs here for b with optimization at l - 1
                            if (m == l - 1) {
                                Fb[l][i][m][r] = alpha[l][i] * As[l][i][r] * Fb[l - 1][r][m][r];
                            } else {
                                sum = 0;
                                for (int j1 = 0; j1 < ns[l]; j1++) {
                                    sum += As[l][i][j1] * Fb[l - 1][j1][m][r];
                                }
                                Fb[l][i][m][r] = sum * alpha[l][i];
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
        double mse = 0.0;
        double rateMod = 1;
        if (ram > 0 && mses != null) {
            mse = util.mse(net.evaluate(x), ans);
            mses[mseIndex] = mse;
            mseIndex = (mseIndex + 1) % ram;
            alltimeMse *= ((double) cycle + 1.0) / ((double) cycle + 2.0);
            alltimeMse += mse / ((double) cycle + 2.0);
            rateMod = expAcc && stochasticBatchSize == 1 ? mse / 0.01 : 1.0;
            if (rateMod < 0.1) {
                rateMod = 0.1;
            } else if (rateMod > 2) {
                rateMod = 2;
            }
        }
        if (momentum > 0) {
            for (int l = 0; l < numLayers - 2; l++) {
                for (int r = 0; r < ns[l + 1]; r++) {
                    for (int j = 0; j < ns[l]; j++) {
                        gradAs[l][r][j] += momentum * lastAs[l][r][j];
                    }
                    gradBs[l][r] += momentum * lastBs[l][r];
                }
            }
            lastAs = gradAs;
            lastBs = gradBs;
        }
        net.gradChange(gradAs, gradBs, scaling * rateMod * trainingRate);
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
