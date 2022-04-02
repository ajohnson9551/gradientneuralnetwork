package core.layer;

import core.ActFunc;

public class ConvolutionalLayerParameters extends LayerParameters {
	public ConvolutionalLayerParameters(int convRadius, int numConvs, int pad, ActFunc actFunc) {
		super(convRadius, numConvs, pad, actFunc);
	}
}
