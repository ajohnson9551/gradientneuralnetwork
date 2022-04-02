package core.network;

import java.io.Serializable;

public abstract class NetworkParameters implements Serializable {

	public final int numInputs;
	public final int numOutputs;

	public NetworkParameters(Integer numInputs, Integer numOutputs) {
		this.numInputs = numInputs;
		this.numOutputs = numOutputs;
	}
}
