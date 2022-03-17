package core;

import java.util.Random;

public class Utility {

    private static Utility utility;
    private static double[] sigmoidValues;
    private static double[] sigmoidPrimeValues;
    private static final int numApproximations = 1000000;
    private static final int approxRange = 150;
    private static final double approxIndexMult = ((double) numApproximations) / (2.0 * ((double) approxRange));

    private Utility() {
        sigmoidValues = new double[numApproximations];
        sigmoidPrimeValues = new double[numApproximations];
        for (int i = 0; i < numApproximations; i++) {
            sigmoidValues[i] = sigmoid(2.0 * approxRange * (i - numApproximations/2.0) / numApproximations);
            sigmoidPrimeValues[i] = sigmoidPrime(2.0 * approxRange * (i - numApproximations/2.0) / numApproximations);
        }
    }

    private double sigmoidPrime(double v) {
        return Math.exp(v / 10.0) / (10.0 * (1 + Math.exp(v / 10.0))*(1 + Math.exp(v / 10.0)));
    }

    public static Utility getUtility() {
        if (utility == null) {
            utility = new Utility();
        }
        return utility;
    }

    public String roundString(double x) {
        String r = x >= 0 ? " " : "";
        r += String.format("%.6f", x);
        return r;
    }

    public void sigmoid(double[] input) {
        for (int i = 0; i < input.length; i++) {
            input[i] = fastSigmoid(input[i]);
        }
    }

    private double sigmoid(double x) {
        return Math.exp(x / 10.0) / (1 + Math.exp(x / 10.0));
    }

    public int indexOfApprox(double x) {
        if (x < -1 * approxRange) {
            return 0;
        } else if (x > approxRange) {
            return numApproximations - 1;
        } else {
            return (int) (approxIndexMult * (x + approxRange));
        }
    }

    public double fastSigmoid(double x) {
        return sigmoidValues[indexOfApprox(x)];
    }

    public double fastSigmoidPrime(double x) {
        return sigmoidPrimeValues[indexOfApprox(x)];
    }

    public double dotProd(double[] a, double[] x) {
        double v = 0;
        for (int i = 0; i < a.length; i++) {
            // in implementation, assume x is more likely to be 0
            // so this should run slightly faster...
            if (x[i] == 0) {
                continue;
            }
            v += a[i] * x[i];
        }
        return v;
    }

    public double[] evaluate(double[][] A, double[] b, double[] x, boolean doSigmoid) {
        double[] y = new double[A.length];
        for (int i = 0; i < A.length; i++) {
            y[i] += dotProd(A[i], x) + b[i];
        }
        if (doSigmoid) {
            sigmoid(y);
        }
        return y;
    }

    public int maxIndex(double[] x) {
        int i = 0;
        double max = -Double.MAX_VALUE;
        for (int j = 0; j < x.length; j++) {
            if (x[j] > max) {
                max = x[j];
                i = j;
            }
        }
        return i;
    }

    public int indexRank(double[] x, int i) {
        double m = x[i];
        int rank = 0;
        for (int j = 0; j < x.length; j++) {
            if (x[j] > m) {
                rank++;
            }
        }
        return rank;
    }

    public void normalize(double[] x) {
        double mag = 0.0;
        for (double d : x) {
            mag += d * d;
        }
        mag = Math.sqrt(mag);
        for (int i = 0; i < x.length; i++) {
            x[i] = x[i] / mag;
        }
    }

    public double mse(double[] guess, double[] correct) {
        double e = 0;
        for (int i = 0; i < guess.length; i++) {
            e += (guess[i] - correct[i]) * (guess[i] - correct[i]);
        }
        return (e / ((double) guess.length));
    }

    public double mse(double[] guess, int correct) {
        double e = 0;
        for (int i = 0; i < guess.length; i++) {
            if (i != correct) {
                e += guess[i] * guess[i];
            } else {
                e += (1 - guess[i]) * (1 - guess[i]);
            }
        }
        return (e / ((double) guess.length));
    }

    public String avgString(double[] a) {
        double avg = 0;
        for (double x : a) {
            avg += x;
        }
        avg = avg / a.length;
        return roundString(avg);
    }
}
