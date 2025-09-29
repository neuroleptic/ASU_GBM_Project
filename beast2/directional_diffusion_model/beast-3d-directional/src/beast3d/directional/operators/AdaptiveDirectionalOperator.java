package beast3d.directional.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import beast.base.math.matrixalgebra.Matrix;

/**
 * Adaptive MCMC operator for directional drift using multivariate proposals
 *
 * This operator learns the correlation structure of the drift parameters
 * and adapts its proposal covariance matrix for efficient sampling.
 *
 */
@Description("Adaptive multivariate operator for directional drift with learned covariance")
public class AdaptiveDirectionalOperator extends Operator {

    final public Input<RealParameter> driftRateInput = new Input<>(
        "driftRate",
        "The drift rate vector parameter to operate on",
        Input.Validate.REQUIRED
    );

    final public Input<Double> scaleFactorInput = new Input<>(
        "scaleFactor",
        "Scale factor for proposal covariance",
        0.01
    );

    final public Input<Integer> burnInInput = new Input<>(
        "burnIn",
        "Number of iterations before starting adaptation",
        1000
    );

    final public Input<Integer> adaptationIntervalInput = new Input<>(
        "adaptationInterval",
        "Interval between covariance updates",
        100
    );

    // Adaptive covariance tracking
    private double[][] proposalCovariance;
    private double[][] empiricalCovariance;
    private double[] meanDrift;
    private double[] sumDrift;
    private double[][] sumDriftProducts;
    private int sampleCount;
    private double scaleFactor;

    // Constants
    private static final double EPSILON = 1e-6; // Small value for numerical stability

    @Override
    public void initAndValidate() {
        scaleFactor = scaleFactorInput.get();

        // Initialize covariance matrices
        proposalCovariance = new double[3][3];
        empiricalCovariance = new double[3][3];
        meanDrift = new double[3];
        sumDrift = new double[3];
        sumDriftProducts = new double[3][3];
        sampleCount = 0;

        // Start with diagonal covariance
        for (int i = 0; i < 3; i++) {
            proposalCovariance[i][i] = scaleFactor;
        }
    }

    @Override
    public double proposal() {
        RealParameter driftRate = driftRateInput.get();

        // Get current values
        double[] currentDrift = new double[3];
        for (int i = 0; i < 3; i++) {
            currentDrift[i] = driftRate.getValue(i);
        }

        // Update empirical statistics
        updateEmpiricalStatistics(currentDrift);

        // Adapt covariance if appropriate
        if (sampleCount > burnInInput.get() &&
            sampleCount % adaptationIntervalInput.get() == 0) {
            updateProposalCovariance();
        }

        // Generate multivariate normal proposal
        double[] proposedDrift = multivariateNormalProposal(currentDrift);

        // Check bounds and apply proposal
        for (int dim = 0; dim < 3; dim++) {
            // Check bounds if they exist
            if (driftRate.getLower() != Double.NEGATIVE_INFINITY ||
                driftRate.getUpper() != Double.POSITIVE_INFINITY) {

                if (proposedDrift[dim] < driftRate.getLower() ||
                    proposedDrift[dim] > driftRate.getUpper()) {
                    // Reject if out of bounds
                    return Double.NEGATIVE_INFINITY;
                }
            }

            // Apply proposal
            driftRate.setValue(dim, proposedDrift[dim]);
        }

        // Symmetric proposal, so Hastings ratio = 0 (log scale)
        return 0.0;
    }

    /**
     * Update empirical statistics for covariance estimation
     */
    private void updateEmpiricalStatistics(double[] drift) {
        sampleCount++;

        // Update sums
        for (int i = 0; i < 3; i++) {
            sumDrift[i] += drift[i];
            for (int j = 0; j < 3; j++) {
                sumDriftProducts[i][j] += drift[i] * drift[j];
            }
        }
    }

    /**
     * Update proposal covariance based on empirical covariance
     */
    private void updateProposalCovariance() {
        if (sampleCount < 2) {
            return;
        }

        // Calculate empirical mean
        for (int i = 0; i < 3; i++) {
            meanDrift[i] = sumDrift[i] / sampleCount;
        }

        // Calculate empirical covariance
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                empiricalCovariance[i][j] = (sumDriftProducts[i][j] / sampleCount) -
                                           (meanDrift[i] * meanDrift[j]);
            }
        }

        double optimalScale = 2.38 * 2.38 / 3.0;
        double alpha = Math.min(0.1, 1.0 / Math.sqrt(sampleCount));

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                proposalCovariance[i][j] = (1 - alpha) * proposalCovariance[i][j] +
                                          alpha * optimalScale * empiricalCovariance[i][j];
            }

            // Add small diagonal component for numerical stability
            proposalCovariance[i][i] += EPSILON;
        }

        // Ensure positive definiteness
        makePositiveDefinite(proposalCovariance);
    }

    /**
     * Generate multivariate normal proposal
     */
    private double[] multivariateNormalProposal(double[] mean) {
        // Cholesky decomposition of covariance matrix
        double[][] L = choleskyDecomposition(proposalCovariance);

        // Generate independent standard normal variables
        double[] z = new double[3];
        for (int i = 0; i < 3; i++) {
            z[i] = Randomizer.nextGaussian();
        }

        // Transform to multivariate normal: x = mean + L * z
        double[] proposal = new double[3];
        for (int i = 0; i < 3; i++) {
            proposal[i] = mean[i];
            for (int j = 0; j <= i; j++) {
                proposal[i] += L[i][j] * z[j];
            }
        }

        return proposal;
    }

    /**
     * Cholesky decomposition for 3x3 matrix
     */
    private double[][] choleskyDecomposition(double[][] A) {
        double[][] L = new double[3][3];

        // Standard Cholesky decomposition algorithm
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0;
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }

                if (i == j) {
                    L[i][j] = Math.sqrt(Math.max(EPSILON, A[i][i] - sum));
                } else {
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }

        return L;
    }

    private void makePositiveDefinite(double[][] matrix) {
        // Simple approach: add small value to diagonal if needed
        double minEigenvalue = estimateMinEigenvalue(matrix);
        if (minEigenvalue < EPSILON) {
            for (int i = 0; i < 3; i++) {
                matrix[i][i] += EPSILON - minEigenvalue;
            }
        }
    }

    /**
     * Estimate minimum eigenvalue using Gershgorin circle theorem
     */
    private double estimateMinEigenvalue(double[][] matrix) {
        double minEigenvalue = Double.POSITIVE_INFINITY;

        for (int i = 0; i < 3; i++) {
            double radius = 0;
            for (int j = 0; j < 3; j++) {
                if (i != j) {
                    radius += Math.abs(matrix[i][j]);
                }
            }
            double lower = matrix[i][i] - radius;
            minEigenvalue = Math.min(minEigenvalue, lower);
        }

        return minEigenvalue;
    }

    @Override
    public void optimize(double logAlpha) {
        // Adapt scale factor based on acceptance rate
        double delta = calcDelta(logAlpha);
        scaleFactor *= Math.exp(delta);

        // Update diagonal elements of proposal covariance
        for (int i = 0; i < 3; i++) {
            proposalCovariance[i][i] *= Math.exp(delta);
        }
    }

    @Override
    public double getCoercableParameterValue() {
        return scaleFactor;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        scaleFactor = value;
    }

    /**
     * Reset adaptation statistics
     */
    public void reset() {
        sampleCount = 0;
        for (int i = 0; i < 3; i++) {
            sumDrift[i] = 0;
            meanDrift[i] = 0;
            for (int j = 0; j < 3; j++) {
                sumDriftProducts[i][j] = 0;
                empiricalCovariance[i][j] = 0;
                proposalCovariance[i][j] = (i == j) ? scaleFactor : 0;
            }
        }
    }

    @Override
    public String getPerformanceSuggestion() {
        double acceptanceRate = m_nNrAccepted / (m_nNrAccepted + m_nNrRejected + 0.0);

        if (acceptanceRate < 0.15) {
            return "Acceptance rate too low for " + getID() + ". Consider decreasing scaleFactor.";
        } else if (acceptanceRate > 0.35) {
            return "Acceptance rate too high for " + getID() + ". Consider increasing scaleFactor.";
        }

        return "";
    }
}