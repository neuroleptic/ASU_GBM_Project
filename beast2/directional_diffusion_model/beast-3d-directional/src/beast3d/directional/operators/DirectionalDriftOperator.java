package beast3d.directional.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

/**
 * MCMC operator for updating directional drift parameters
 *
 * This operator performs a random walk on the 3D drift vector,
 * allowing efficient exploration of directional preferences in spatial evolution.
 *
 */
@Description("Random walk operator for directional drift parameters in 3D spatial evolution")
public class DirectionalDriftOperator extends Operator {

    final public Input<RealParameter> driftRateInput = new Input<>(
        "driftRate",
        "The drift rate vector parameter to operate on",
        Input.Validate.REQUIRED
    );

    final public Input<Double> windowSizeInput = new Input<>(
        "windowSize",
        "Size of the random walk window for drift updates",
        0.1
    );

    final public Input<Boolean> optimizeWindowSizeInput = new Input<>(
        "optimizeWindowSize",
        "Whether to automatically optimize window size for target acceptance rate",
        true
    );

    final public Input<Boolean> uniformlyDistributedInput = new Input<>(
        "uniformlyDistributed",
        "Use uniform distribution instead of normal for proposals",
        false
    );

    final public Input<Boolean> updateAllDimensionsInput = new Input<>(
        "updateAllDimensions",
        "Update all three dimensions simultaneously (true) or one at a time (false)",
        false
    );

    // Operator tuning
    private double windowSize;
    private static final double TARGET_ACCEPTANCE = 0.234; // Optimal for random walk

    @Override
    public void initAndValidate() {
        windowSize = windowSizeInput.get();

        // Validate drift rate dimension
        if (driftRateInput.get().getDimension() != 3) {
            throw new IllegalArgumentException("Drift rate must be 3-dimensional");
        }
    }

    @Override
    public double proposal() {
        RealParameter driftRate = driftRateInput.get();

        double hastingsRatio = 0.0;

        if (updateAllDimensionsInput.get()) {
            // Update all dimensions simultaneously
            hastingsRatio = proposeAllDimensions(driftRate);
        } else {
            // Update one dimension at a time
            hastingsRatio = proposeOneDimension(driftRate);
        }

        return hastingsRatio;
    }

    /**
     * Propose update to all three dimensions simultaneously
     */
    private double proposeAllDimensions(RealParameter driftRate) {
        double[] oldValues = new double[3];
        double[] newValues = new double[3];

        // Store old values and propose new ones
        for (int dim = 0; dim < 3; dim++) {
            oldValues[dim] = driftRate.getValue(dim);

            if (uniformlyDistributedInput.get()) {
                // Uniform random walk
                double delta = Randomizer.uniform(-windowSize, windowSize);
                newValues[dim] = oldValues[dim] + delta;
            } else {
                // Gaussian random walk
                double delta = Randomizer.nextGaussian() * windowSize;
                newValues[dim] = oldValues[dim] + delta;
            }

            // Check bounds if they exist
            if (driftRate.getLower() != Double.NEGATIVE_INFINITY ||
                driftRate.getUpper() != Double.POSITIVE_INFINITY) {

                if (newValues[dim] < driftRate.getLower() ||
                    newValues[dim] > driftRate.getUpper()) {
                    // Reflection at boundary
                    if (newValues[dim] < driftRate.getLower()) {
                        newValues[dim] = 2 * driftRate.getLower() - newValues[dim];
                    }
                    if (newValues[dim] > driftRate.getUpper()) {
                        newValues[dim] = 2 * driftRate.getUpper() - newValues[dim];
                    }

                    // If still out of bounds, reject
                    if (newValues[dim] < driftRate.getLower() ||
                        newValues[dim] > driftRate.getUpper()) {
                        return Double.NEGATIVE_INFINITY;
                    }
                }
            }
        }

        // Apply the proposal
        for (int dim = 0; dim < 3; dim++) {
            driftRate.setValue(dim, newValues[dim]);
        }

        // Symmetric proposal, so Hastings ratio = 0 (log scale)
        return 0.0;
    }

    /**
     * Propose update to one dimension at a time
     */
    private double proposeOneDimension(RealParameter driftRate) {
        // Randomly select dimension to update
        int dim = Randomizer.nextInt(3);

        double oldValue = driftRate.getValue(dim);
        double newValue;

        if (uniformlyDistributedInput.get()) {
            // Uniform random walk
            double delta = Randomizer.uniform(-windowSize, windowSize);
            newValue = oldValue + delta;
        } else {
            // Gaussian random walk
            double delta = Randomizer.nextGaussian() * windowSize;
            newValue = oldValue + delta;
        }

        // Check bounds
        if (driftRate.getLower() != Double.NEGATIVE_INFINITY ||
            driftRate.getUpper() != Double.POSITIVE_INFINITY) {

            if (newValue < driftRate.getLower() || newValue > driftRate.getUpper()) {
                // Reflection at boundary
                if (newValue < driftRate.getLower()) {
                    newValue = 2 * driftRate.getLower() - newValue;
                }
                if (newValue > driftRate.getUpper()) {
                    newValue = 2 * driftRate.getUpper() - newValue;
                }

                // If still out of bounds, reject
                if (newValue < driftRate.getLower() || newValue > driftRate.getUpper()) {
                    return Double.NEGATIVE_INFINITY;
                }
            }
        }

        // Apply the proposal
        driftRate.setValue(dim, newValue);

        // Symmetric proposal, so Hastings ratio = 0 (log scale)
        return 0.0;
    }

    @Override
    public void optimize(double logAlpha) {
        if (!optimizeWindowSizeInput.get()) {
            return;
        }

        // Adjust window size based on acceptance rate
        double delta = calcDelta(logAlpha);

        // Prevent window size from becoming too small or too large
        double newWindowSize = windowSize * Math.exp(delta);
        if (newWindowSize > 0 && newWindowSize < 10) {
            windowSize = newWindowSize;
        }
    }

    @Override
    public double getCoercableParameterValue() {
        return windowSize;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        windowSize = value;
    }

    @Override
    public String getPerformanceSuggestion() {
        double acceptanceRate = m_nNrAccepted / (m_nNrAccepted + m_nNrRejected + 0.0);

        if (acceptanceRate < 0.15) {
            return "Try decreasing windowSize for " + getID();
        } else if (acceptanceRate > 0.35) {
            return "Try increasing windowSize for " + getID();
        }

        return "";
    }
}