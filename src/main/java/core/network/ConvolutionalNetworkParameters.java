package core.network;

import core.layer.Layer;
import core.layer.LayerParameters;

import java.util.List;

public class ConvolutionalNetworkParameters extends NetworkParameters {

	Layer[] layers;
	int batchSize;

	public ConvolutionalNetworkParameters(int[] inputSize, int numOutputs, List<LayerParameters> layerParams, int batchSize) {
		super(inputSize[0] * inputSize[1] * inputSize[2], numOutputs);
		this.layers = new Layer[layerParams.size()];

		this.layers[0] = layerParams.get(0).makeLayer(inputSize);

		for (int i = 1; i < this.layers.length; i++) {
			this.layers[i] = layerParams.get(i).makeLayer(this.layers[i - 1].layerParam.outputSize);
		}

		this.batchSize = batchSize;
	}
}
