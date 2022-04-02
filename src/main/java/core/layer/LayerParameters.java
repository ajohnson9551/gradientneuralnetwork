package core.layer;

import core.ActFunc;

import java.io.Serializable;

public abstract class LayerParameters implements Serializable {

	public int[] inputSize;
	public int[] outputSize;

	public final ActFunc actFunc;

	public final int poolSize;
	public final int stride;

	public final int convRadius;
	public final int numConvs;
	public final int pad;
	public final int convMod;

	public final LayerType layerType;

	public LayerParameters(int outputLength, ActFunc actFunc) {
		assert outputLength > 0;
		this.outputSize = new int[]{outputLength, 1, 1};
		this.actFunc = actFunc;
		this.poolSize = 0;
		this.convRadius = 0;
		this.numConvs = 0;
		this.stride = 0;
		this.pad = 0;
		this.convMod = 0;
		this.layerType = LayerType.FULL;
	}

	public LayerParameters(int poolSize, int stride) {
		assert stride > 0;
		assert poolSize > 0;
		this.actFunc = null;
		this.poolSize = poolSize;
		this.convRadius = 0;
		this.numConvs = 0;
		this.pad = 0;
		this.stride = stride;
		this.convMod = 0;
		this.layerType = LayerType.POOL;
	}

	public LayerParameters(int convRadius, int numConvs, int pad, ActFunc actFunc) {
		assert convRadius > 0;
		assert numConvs > 0;
		assert pad >= 0;
		this.actFunc = actFunc;
		this.poolSize = 0;
		this.convRadius = convRadius;
		this.numConvs = numConvs;
		this.pad = pad;
		this.stride = 0;
		this.convMod = Math.min(0, 1 + pad - convRadius);
		this.layerType = LayerType.CONV;
	}

	public int[] getOutputSize(int[] inputSize) {
		return switch (layerType) {
			case FULL -> this.outputSize; // full layer should already know this
			case POOL -> new int[]{inputSize[0] / stride, inputSize[1] / stride, inputSize[2]};
			case CONV -> new int[]{inputSize[0] + (2 * convMod), inputSize[1] + (2 * convMod), inputSize[2] * numConvs};
		};
	}

	public Layer makeLayer(int[] inputSize, int[] outputSize) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		return switch (layerType) {
			case FULL -> new FullLayer(this);
			case POOL -> new PoolLayer(this);
			case CONV -> new ConvolutionalLayer(this);
		};
	}

	public Layer makeLayer(int[] inputSize) {
		return this.makeLayer(inputSize, this.getOutputSize(inputSize));
	}
}
