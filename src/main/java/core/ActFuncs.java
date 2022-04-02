package core;

import java.util.Arrays;

public final class ActFuncs {

	private static double[] sigmoidValues;
	private static double[] sigmoidPrimeValues;
	private static final int numApproximations = 1000000;
	private static final int approxRange = 150;
	private static final double approxIndexMult = ((double) numApproximations) / (2.0 * ((double) approxRange));

	private static ActFuncs actFuncs;

	private ActFuncs() {
		sigmoidValues = new double[numApproximations];
		sigmoidPrimeValues = new double[numApproximations];
		for (int i = 0; i < numApproximations; i++) {
			sigmoidValues[i] = slowSigmoid(2.0 * approxRange * (i - numApproximations/2.0) / numApproximations);
			sigmoidPrimeValues[i] = slowSigmoidPrime(2.0 * approxRange * (i - numApproximations/2.0) / numApproximations);
		}
	}

	public static ActFuncs getActFuncs() {
		if (actFuncs == null) {
			actFuncs = new ActFuncs();
		}
		return actFuncs;
	}

	public double actFunc(double x, ActFunc actFunc) {
		return switch (actFunc) {
			case SIGMOID -> sigmoid(x);
			case RELU -> relu(x);
			case IDENTITY -> identity(x);
		};
	}

	public double actFuncPrime(double x, ActFunc actFunc) {
		return switch (actFunc) {
			case SIGMOID -> sigmoidPrime(x);
			case RELU -> reluPrime(x);
			case IDENTITY -> identityPrime();
		};
	}

	public void actFuncify(double[] w, ActFunc actFunc) {
		switch (actFunc) {
			case SIGMOID -> sigmoid(w);
			case RELU -> relu(w);
			case IDENTITY -> identify();
		}
	}

	public void actFuncPrimeify(double[] x, ActFunc actFunc) {
		switch (actFunc) {
			case SIGMOID -> sigmoidPrime(x);
			case RELU -> reluPrime(x);
			case IDENTITY -> identityPrime(x);
		}
	}

	private double slowSigmoid(double x) {
		return Math.exp(x / 10.0) / (1 + Math.exp(x / 10.0));
	}

	private double slowSigmoidPrime(double v) {
		return Math.exp(v / 10.0) / (10.0 * (1 + Math.exp(v / 10.0))*(1 + Math.exp(v / 10.0)));
	}

	private double relu(double x) {
		return x > 0 ? x : 0;
	}

	private double reluPrime(double x) {
		return x > 0 ? 1 : 0;
	}

	private void sigmoid(double[] input) {
		for (int i = 0; i < input.length; i++) {
			input[i] = sigmoid(input[i]);
		}
	}

	private void relu(double[] input) {
		for (int i = 0; i < input.length; i++) {
			input[i] = relu(input[i]);
		}
	}

	private void sigmoidPrime(double[] input) {
		for (int i = 0; i < input.length; i++) {
			input[i] = sigmoidPrime(input[i]);
		}
	}

	private void reluPrime(double[] input) {
		for (int i = 0; i < input.length; i++) {
			input[i] = reluPrime(input[i]);
		}
	}

	private double identity(double x) {
		return x;
	}

	private double identityPrime() {
		return 1;
	}

	private void identify() {
	}

	private void identityPrime(double[] input) {
		Arrays.fill(input, 1);
	}

	private double sigmoid(double x) {
		return sigmoidValues[indexOfApprox(x)];
	}

	private double sigmoidPrime(double x) {
		return sigmoidPrimeValues[indexOfApprox(x)];
	}

	private int indexOfApprox(double x) {
		if (x < -1 * approxRange) {
			return 0;
		} else if (x > approxRange) {
			return numApproximations - 1;
		} else {
			return (int) (approxIndexMult * (x + approxRange));
		}
	}
}
