class HMM(numStates: Int, numSymbols: Int) {
    private val A = Array(numStates) { FloatArray(numStates) } // Transition matrix
    private val B = Array(numStates) { FloatArray(numSymbols) } // Emission matrix
    private val pi = FloatArray(numStates) // Initial state probabilities

    init {
        // Initialize A, B, and pi
        for (i in 0 until numStates) {
            pi[i] = 1.0f / numStates
            for (j in 0 until numStates) {
                A[i][j] = 1.0f / numStates
            }
            for (j in 0 until numSymbols) {
                B[i][j] = 1.0f / numSymbols
            }
        }
    }

    fun forward(obs: IntArray): Float {
        // Compute the forward probability of the observations
        val T = obs.size
        val alpha = Array(T) { FloatArray(numStates) }

        // Initialize alpha
        for (i in 0 until numStates) {
            alpha[0][i] = pi[i] * B[i][obs[0]]
        }

        // Recursively compute alpha
        for (t in 1 until T) {
            for (j in 0 until numStates) {
                alpha[t][j] = 0.0f
                for (i in 0 until numStates) {
                    alpha[t][j] += alpha[t-1][i] * A[i][j]
                }
                alpha[t][j] *= B[j][obs[t]]
            }
        }

        // Return the sum of the final alpha values
        return alpha[T-1].sum()
    }
}
