package core.web;

import core.layer.LayerType;
import core.layer.PoolType;

public class WPoolLayer extends WLayer {
	public final int stride;
	public final int poolSize;
	public final PoolType poolType;

	public WPoolLayer(int[] inputSize, int[] outputSize, int stride, int poolSize, PoolType poolType) {
		super(inputSize, outputSize, LayerType.POOL);
		this.stride = stride;
		this.poolSize = poolSize;
		this.poolType = poolType;
	}
}
