package core.web;

import core.ActFunc;
import core.layer.LayerType;

public class WConvolutionalLayer extends WLayer {
	public final double[][][] Cs;
	public final int pad;
	public final ActFunc actFunc;

	public WConvolutionalLayer(int[] inputSize, int[] outputSize, double[][][] Cs, int pad, ActFunc actFunc) {
		super(inputSize, outputSize, LayerType.CONV);
		this.Cs = Cs;
		this.pad = pad;
		this.actFunc = actFunc;
	}
}
