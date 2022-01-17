package core;

import java.util.Random;

public class LayeredNetworkParameters extends NetworkParameters {

    protected int[] hiddenLayerSizes;

    public LayeredNetworkParameters(int[] hiddenLayerSizes, int numInputs, int numOutputs) {
        super(numInputs, numOutputs);
        this.hiddenLayerSizes = hiddenLayerSizes;
    }

    public double[][][] setupWeights(boolean zero) {
        int l = hiddenLayerSizes.length;
        double[][][] weights = new double[l + 1][][];
        if (l == 0) {
            weights[0] = initialMatrix(numOutputs, numInputs, zero);
        } else {
            weights[0] = initialMatrix(hiddenLayerSizes[0], numInputs, zero);
            for (int i = 1; i < l; i++) {
                weights[i] = initialMatrix(hiddenLayerSizes[i], hiddenLayerSizes[i - 1], zero);
            }
            weights[l] = initialMatrix(numOutputs, hiddenLayerSizes[l - 1], zero);
        }
        return weights;
    }

    public double[][] setupOffsets(boolean zero) {
        int l = hiddenLayerSizes.length;
        double[][] offsets = new double[l + 1][];
        if (l == 0) {
            offsets[0] = initialColumn(numOutputs, zero);
        } else {
            offsets[0] = initialColumn(hiddenLayerSizes[0], zero);
            for (int i = 1; i < l; i++) {
                offsets[i] = initialColumn(hiddenLayerSizes[i], zero);
            }
            offsets[l] = initialColumn(numOutputs, zero);
        }
        return offsets;
    }

    public double[] initialColumn(int rows, boolean zero) {
        double[][] colMat = initialMatrix(rows, 1, zero);
        double[] col = new double[rows];
        for (int i = 0; i < rows; i++) {
            col[i] = colMat[i][0];
        }
        return col;
    }

    public double[][] initialMatrix(int rows, int columns, boolean zero) {
        // can randomize the zero matrix here if desired
        double[][] out = zeroMatrix(rows, columns);
        for (double[] col : out) {
            for (int i = 0; i < col.length; i++) {
                col[i] = zero ? 0.0 : initialValue();
            }
        }
        return out;
    }

    public double initialValue() {
        Random rand = new Random();
        return 2 * rand.nextDouble() - 1;
    }

    public double[][] zeroMatrix(int rows, int columns) {
        return new double[rows][columns];
    }
}
