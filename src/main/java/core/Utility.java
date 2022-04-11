package core;

import java.util.Random;

public final class Utility {

	static Random rand = new Random();

	private Utility() {}

	public static double randVal(double center, double radius) {
		return (2 * radius * rand.nextDouble() - radius) + center;
	}

	public static String roundString(double x) {
		String r = x >= 0 ? " " : "";
		r += String.format("%.6f", x);
		return r;
	}

	public static double dotProd(double[] a, double[] x) {
		double v = 0;
		for (int i = 0; i < a.length; i++) {
			if (x[i] == 0) {
				continue;
			}
			v += a[i] * x[i];
		}
		return v;
	}

	public static double[] evaluate(double[][] A, double[] b, double[] x, ActFunc actFunc) {
		double[] y = new double[A.length];
		for (int i = 0; i < A.length; i++) {
			y[i] += dotProd(A[i], x) + b[i];
		}
		if (actFunc != null) {
			ActFuncs.getActFuncs().actFuncify(y, actFunc);
		}
		return y;
	}

	public static int maxIndex(double[] x) {
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

	public static double getOrDefault(double[][][] x, int i, int j, int k, double def) {
		if (i < 0 || i >= x.length || j < 0 || j >= x[0].length || k < 0 || k >= x[0][0].length) {
			return def;
		}
		return x[i][j][k];
	}

	public static void setIfCan(double[][][] x, int i, int j, int k, double value) {
		if (i < 0 || i >= x.length || j < 0 || j >= x[0].length || k < 0 || k >= x[0][0].length) {
			return;
		}
		x[i][j][k] = value;
	}

	public static void setIfCan(double[][][] x, int i, int j, double[] arr) {
		if (i < 0 || i >= x.length || j < 0 || j >= x[0].length) {
			return;
		}
		x[i][j] = arr;
	}

	public static double mse(double[] guess, double[] correct) {
		double e = 0;
		for (int i = 0; i < guess.length; i++) {
			e += (guess[i] - correct[i]) * (guess[i] - correct[i]);
		}
		return (e / ((double) guess.length));
	}

	public static double mse(double[] guess, int correct) {
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

	public static String avgString(double[] a) {
		double avg = 0;
		for (double x : a) {
			avg += x;
		}
		avg = avg / a.length;
		return roundString(avg);
	}
}
