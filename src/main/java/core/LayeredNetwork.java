package core;

import java.util.Arrays;

public class LayeredNetwork implements Network {

    private LayeredNetworkParameters params;
    private double[][][] weights;
    private double[][] offsets;

    private Utility util = Utility.getUtility();


    public LayeredNetwork(LayeredNetworkParameters params) {
        this.params = params;
        this.weights = params.setupWeights(false);
        this.offsets = params.setupOffsets(false);
    }

    @Override
    public double[] evaluate(double[] x) {
        return evaluate(x, weights.length);
    }

    public double[] evaluate(double[] x, int toLayer) {
        if (toLayer == 0) {
            return Arrays.copyOf(x, x.length);
        }
        double[] result;
        result = util.evaluate(weights[0], offsets[0], x, true);
        for (int i = 1; i < Math.min(toLayer, weights.length); i++) {
            result = util.evaluate(weights[i], offsets[i], result, true);
        }
        return result;
    }

    public void gradChange(double[][][] gradAs, double[][] gradBs, double trainingRate) {
        for (int v = 0; v < gradAs.length; v++) {
            for (int i = 0; i < gradAs[v].length; i++) {
                for (int j = 0; j < gradAs[v][0].length; j++) {
                    weights[v][i][j] = weights[v][i][j] - (trainingRate * gradAs[v][i][j]);
                }
                offsets[v][i] = offsets[v][i]- (trainingRate * gradBs[v][i]);
            }
        }
    }

    public double[][][] getWeights() {
        return weights;
    }

    public double[][] getOffsets() {
        return offsets;
    }

    public int getNumHiddenLayers() {
        return params.hiddenLayerSizes.length;
    }

    public double[][][] getEmptyGradAs() {
        return params.setupWeights(true);
    }

    public double[][] getEmptyGradBs() {
        return params.setupOffsets(true);
    }
}
