package core.web;

import core.layer.LayerType;

import java.io.Serializable;

public abstract class WLayer implements Serializable {
	public final int[] inputSize;
	public final int[] outputSize;
	public final LayerType layerType;

	public WLayer(int[] inputSize, int[] outputSize, LayerType layerType) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.layerType = layerType;
	}
}
