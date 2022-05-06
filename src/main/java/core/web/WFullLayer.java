package core.web;

import core.ActFunc;
import core.layer.LayerType;

public class WFullLayer extends WLayer {
	public final double[][] A;
	public final double[] b;
	public final ActFunc actFunc;

	public WFullLayer(int[] inputSize, int[] outputSize, double[][] A, double[] b, ActFunc actFunc) {
		super(inputSize, outputSize, LayerType.FULL);
		this.A = A;
		this.b = b;
		this.actFunc = actFunc;
	}
}
