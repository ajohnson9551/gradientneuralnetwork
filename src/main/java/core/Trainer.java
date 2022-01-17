package core;

import java.util.Random;

public class Trainer {

    private final double trainingRate;
    private final Network net;
    private final Fitness trainFit;
    private final double percentToDo;
    private final boolean stochastic;

    private final Random rand = new Random();
    private final Utility util = Utility.getUtility();

    public Trainer(double trainingRate, Network net, Fitness trainFit, double percentToDo, boolean stochastic) {
        this.trainingRate = trainingRate;
        this.net = net;
        this.trainFit = trainFit;
        this.percentToDo = percentToDo;
        this.stochastic = stochastic;
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
                if (((LayeredNetwork) net).getNumHiddenLayers() == 1) {
                    trainStep((LayeredNetwork) net);
                }
            } else {
                System.out.println("Error! Unable to train this kind of network yet.");
                System.exit(0);
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
        int toDo = (int) (data.length * percentToDo);

        double[] c = new double[m];
        double[] q = new double[m];
        double[] e = new double[m];
        double[] alpha;
        double[] r = new double[n];
        double[] d = new double[n];
        double[] w = new double[n];
        double[][] z = new double[n][m];

        for (int k = 0; k < toDo; k++) {
            if (stochastic) {
                k = rand.nextInt(toDo);
            }
            double[] x = data[k];
            alpha = net.evaluate(x, 1);
            for (int i = 0; i < n; i++) {
                r[i] = util.dotProd(As[1][i], alpha) + bs[1][i];
                d[i] = util.fastSigmoid(r[i]) - answers[k][i];
                w[i] = util.fastSigmoidPrime(r[i]) * d[i];
            }
            for (int j = 0; j < m; j++) {
                c[j] = util.dotProd(As[0][j], x) + bs[0][j];
                q[j] = util.fastSigmoid(c[j]);
                e[j] = util.fastSigmoidPrime(c[j]);
                for (int i = 0; i < n; i++) {
                    z[i][j] = As[1][i][j] * e[j];
                }

            }
            double scaling = stochastic ? (double) n : (double) (toDo * n);
            scaling = 2.0 / scaling;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    for (int k1 = 0; k1 < u; k1++) {
                        gradAs[0][j][k1] += scaling * w[i] * z[i][j] * x[k1];
                    }
                    gradAs[1][i][j] += scaling * w[i] * q[j];
                    gradBs[0][j] += scaling * w[i] * z[i][j];
                }
                gradBs[1][i] += scaling * w[i];
            }
            net.gradChange(gradAs, gradBs, trainingRate);
            if (stochastic) {
                break;
            }
        }
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
