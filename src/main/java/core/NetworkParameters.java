package core;

public abstract class NetworkParameters {

    protected final int numInputs;
    protected final int numOutputs;

    public NetworkParameters(Integer numInputs, Integer numOutputs) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
    }
}
