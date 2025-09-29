package beast3d.directional.model;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.CalculationNode;
import beast.base.inference.parameter.RealParameter;
import beast.base.evolution.tree.Node;

/**
 * Directional Diffusion Model for 3D spatial phylogenetics
 *
 * This model extends standard Brownian motion by adding a directional drift component,
 * allowing for preferential movement in certain directions during evolution.
 *
 * Mathematical model:
 * X(t) ~ Normal(X(0) + μt, Σt)
 * where μ is the drift vector and Σ is the diffusion matrix
 *
 */
@Description("Directional diffusion model with drift for 3D spatial evolution")
public class DirectionalDiffusionModel extends CalculationNode {

    final public Input<RealParameter> driftRateInput = new Input<>(
        "driftRate",
        "Directional drift rate vector (μx, μy, μz) representing preferential movement direction",
        Input.Validate.REQUIRED
    );

    final public Input<RealParameter> diffusionRateInput = new Input<>(
        "diffusionRate",
        "Isotropic diffusion rate (σ²) controlling random spread magnitude",
        Input.Validate.REQUIRED
    );

    final public Input<RealParameter> precisionMatrixInput = new Input<>(
        "precisionMatrix",
        "3x3 precision matrix (inverse covariance) for multivariate normal distribution. " +
        "If not specified, uses diagonal matrix with diffusionRate",
        Input.Validate.OPTIONAL
    );

    final public Input<Boolean> allowBranchSpecificDriftInput = new Input<>(
        "allowBranchSpecificDrift",
        "Allow different drift vectors for different branches/clades",
        false
    );

    final public Input<RealParameter> branchDriftIndicatorsInput = new Input<>(
        "branchDriftIndicators",
        "Binary indicators (0/1) for branch-specific drift activation",
        Input.Validate.OPTIONAL
    );

    // Cached values
    private double[] driftRate;
    private double diffusionRate;
    private double[][] precisionMatrix;
    private double[][] varianceMatrix;
    private boolean needsUpdate = true;

    @Override
    public void initAndValidate() {
        // Validate drift rate dimension
        if (driftRateInput.get().getDimension() != 3) {
            throw new IllegalArgumentException("Drift rate must be 3-dimensional (x, y, z)");
        }

        // Validate precision matrix if provided
        if (precisionMatrixInput.get() != null) {
            int dim = precisionMatrixInput.get().getDimension();
            if (dim != 9 && dim != 3) {
                throw new IllegalArgumentException(
                    "Precision matrix must be either 3x3 (9 values) or diagonal (3 values)"
                );
            }
        }

        // Initialize matrices
        precisionMatrix = new double[3][3];
        varianceMatrix = new double[3][3];

        updateCachedValues();
    }

    /**
     * Calculate expected drift over a branch of given length
     *
     * @param branchLength The length of the branch
     * @return 3D vector of expected displacement due to drift
     */
    public double[] calculateDrift(double branchLength) {
        if (needsUpdate) {
            updateCachedValues();
        }

        double[] drift = new double[3];
        for (int i = 0; i < 3; i++) {
            drift[i] = driftRate[i] * branchLength;
        }
        return drift;
    }

    /**
     * Calculate drift for a specific branch (allows branch-specific drift)
     *
     * @param node The tree node representing the branch
     * @return 3D vector of expected displacement due to drift
     */
    public double[] calculateDriftForBranch(Node node) {
        if (!allowBranchSpecificDriftInput.get()) {
            return calculateDrift(node.getLength());
        }

        // Branch-specific drift logic
        if (branchDriftIndicatorsInput.get() != null) {
            int nodeIndex = node.getNr();
            if (nodeIndex < branchDriftIndicatorsInput.get().getDimension()) {
                double indicator = branchDriftIndicatorsInput.get().getValue(nodeIndex);
                if (indicator == 0.0) {
                    // No drift for this branch
                    return new double[]{0.0, 0.0, 0.0};
                }
            }
        }

        return calculateDrift(node.getLength());
    }

    /**
     * Calculate variance-covariance matrix for a branch
     *
     * @param branchLength The length of the branch
     * @return 3x3 variance-covariance matrix
     */
    public double[][] calculateVariance(double branchLength) {
        if (needsUpdate) {
            updateCachedValues();
        }

        double[][] variance = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                variance[i][j] = varianceMatrix[i][j] * branchLength;
            }
        }
        return variance;
    }

    /**
     * Get the precision matrix (inverse of variance-covariance matrix)
     *
     * @return 3x3 precision matrix
     */
    public double[][] getPrecisionMatrix() {
        if (needsUpdate) {
            updateCachedValues();
        }
        return precisionMatrix;
    }

    /**
     * Calculate log determinant of precision matrix
     * Used in likelihood calculations
     *
     * @return Natural log of determinant of precision matrix
     */
    public double getLogDetPrecision() {
        if (needsUpdate) {
            updateCachedValues();
        }

        // For diagonal matrix, log det is sum of log of diagonal elements
        if (isDiagonal()) {
            double logDet = 0.0;
            for (int i = 0; i < 3; i++) {
                logDet += Math.log(precisionMatrix[i][i]);
            }
            return logDet;
        }

        // For full matrix, compute determinant
        return Math.log(determinant3x3(precisionMatrix));
    }

    /**
     * Calculate quadratic form: (x - μ)ᵀ P (x - μ)
     * Central to multivariate normal likelihood
     *
     * @param difference Vector difference (x - μ)
     * @return Quadratic form result
     */
    public double calculateQuadraticForm(double[] difference) {
        if (needsUpdate) {
            updateCachedValues();
        }

        double result = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result += difference[i] * precisionMatrix[i][j] * difference[j];
            }
        }
        return result;
    }

    /**
     * Update cached values when parameters change
     */
    private void updateCachedValues() {
        // Update drift rate
        driftRate = new double[3];
        for (int i = 0; i < 3; i++) {
            driftRate[i] = driftRateInput.get().getValue(i);
        }

        // Update diffusion rate
        diffusionRate = diffusionRateInput.get().getValue();

        // Update precision matrix
        if (precisionMatrixInput.get() != null) {
            RealParameter precision = precisionMatrixInput.get();
            if (precision.getDimension() == 3) {
                // Diagonal precision matrix
                for (int i = 0; i < 3; i++) {
                    precisionMatrix[i][i] = precision.getValue(i);
                    varianceMatrix[i][i] = 1.0 / precision.getValue(i);
                }
            } else {
                // Full 3x3 precision matrix
                int idx = 0;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        precisionMatrix[i][j] = precision.getValue(idx++);
                    }
                }
                // Calculate variance matrix as inverse of precision
                varianceMatrix = invert3x3(precisionMatrix);
            }
        } else {
            // Use isotropic diffusion rate
            for (int i = 0; i < 3; i++) {
                precisionMatrix[i][i] = 1.0 / diffusionRate;
                varianceMatrix[i][i] = diffusionRate;
            }
        }

        needsUpdate = false;
    }

    /**
     * Check if precision matrix is diagonal
     */
    private boolean isDiagonal() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i != j && Math.abs(precisionMatrix[i][j]) > 1e-10) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Calculate determinant of 3x3 matrix
     */
    private double determinant3x3(double[][] m) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    /**
     * Invert 3x3 matrix
     */
    private double[][] invert3x3(double[][] m) {
        double det = determinant3x3(m);
        if (Math.abs(det) < 1e-10) {
            throw new IllegalArgumentException("Singular matrix cannot be inverted");
        }

        double[][] inv = new double[3][3];
        inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
        inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
        inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
        inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
        inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
        inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
        inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
        inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
        inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;

        return inv;
    }

    @Override
    protected void store() {
        needsUpdate = true;
        super.store();
    }

    @Override
    protected void restore() {
        needsUpdate = true;
        super.restore();
    }

    @Override
    protected boolean requiresRecalculation() {
        needsUpdate = true;
        return true;
    }
}