package core;

import java.io.Serializable;

public abstract class NetworkParameters implements Serializable {

	protected final int numInputs;
	protected final int numOutputs;

	public NetworkParameters(Integer numInputs, Integer numOutputs) {
		this.numInputs = numInputs;
		this.numOutputs = numOutputs;
	}
}
