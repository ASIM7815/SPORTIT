// Custom Regression Models Implementation using Pure JavaScript (NumPy equivalent)

class Matrix {
    static multiply(a, b) {
        const result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < b[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < a[0].length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    static transpose(matrix) {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }

    static inverse(matrix) {
        const n = matrix.length;
        const identity = Array(n).fill().map((_, i) => 
            Array(n).fill().map((_, j) => i === j ? 1 : 0)
        );
        
        const augmented = matrix.map((row, i) => [...row, ...identity[i]]);
        
        for (let i = 0; i < n; i++) {
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
            
            const pivot = augmented[i][i];
            if (Math.abs(pivot) < 1e-10) continue;
            
            for (let j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    const factor = augmented[k][i];
                    for (let j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        return augmented.map(row => row.slice(n));
    }

    static add(a, b) {
        return a.map((row, i) => row.map((val, j) => val + b[i][j]));
    }

    static scalarMultiply(matrix, scalar) {
        return matrix.map(row => row.map(val => val * scalar));
    }
}

class LinearRegression {
    constructor() {
        this.weights = null;
        this.bias = null;
    }

    fit(X, y) {
        const n = X.length;
        const ones = Array(n).fill([1]);
        const X_b = X.map((row, i) => [1, ...row]);
        
        const X_T = Matrix.transpose(X_b);
        const X_T_X = Matrix.multiply(X_T, X_b);
        const X_T_X_inv = Matrix.inverse(X_T_X);
        const X_T_y = Matrix.multiply(X_T, y.map(val => [val]));
        const theta = Matrix.multiply(X_T_X_inv, X_T_y);
        
        this.bias = theta[0][0];
        this.weights = theta.slice(1).map(w => w[0]);
    }

    predict(X) {
        return X.map(row => {
            let sum = this.bias;
            for (let i = 0; i < row.length; i++) {
                sum += row[i] * this.weights[i];
            }
            return sum;
        });
    }
}

class RidgeRegression {
    constructor(alpha = 1.0) {
        this.alpha = alpha;
        this.weights = null;
        this.bias = null;
    }

    fit(X, y) {
        const n = X.length;
        const X_b = X.map((row, i) => [1, ...row]);
        
        const X_T = Matrix.transpose(X_b);
        const X_T_X = Matrix.multiply(X_T, X_b);
        
        const identity = Array(X_b[0].length).fill().map((_, i) => 
            Array(X_b[0].length).fill().map((_, j) => i === j ? this.alpha : 0)
        );
        identity[0][0] = 0;
        
        const regularized = Matrix.add(X_T_X, identity);
        const regularized_inv = Matrix.inverse(regularized);
        const X_T_y = Matrix.multiply(X_T, y.map(val => [val]));
        const theta = Matrix.multiply(regularized_inv, X_T_y);
        
        this.bias = theta[0][0];
        this.weights = theta.slice(1).map(w => w[0]);
    }

    predict(X) {
        return X.map(row => {
            let sum = this.bias;
            for (let i = 0; i < row.length; i++) {
                sum += row[i] * this.weights[i];
            }
            return sum;
        });
    }
}

class LassoRegression {
    constructor(alpha = 1.0, maxIter = 1000, tol = 1e-4) {
        this.alpha = alpha;
        this.maxIter = maxIter;
        this.tol = tol;
        this.weights = null;
        this.bias = null;
    }

    softThreshold(x, lambda) {
        if (x > lambda) return x - lambda;
        if (x < -lambda) return x + lambda;
        return 0;
    }

    fit(X, y) {
        const n = X.length;
        const m = X[0].length;
        
        this.weights = Array(m).fill(0);
        this.bias = y.reduce((a, b) => a + b, 0) / n;
        
        for (let iter = 0; iter < this.maxIter; iter++) {
            const weightsOld = [...this.weights];
            
            for (let j = 0; j < m; j++) {
                let residual = 0;
                let norm = 0;
                
                for (let i = 0; i < n; i++) {
                    let pred = this.bias;
                    for (let k = 0; k < m; k++) {
                        if (k !== j) pred += X[i][k] * this.weights[k];
                    }
                    residual += X[i][j] * (y[i] - pred);
                    norm += X[i][j] * X[i][j];
                }
                
                if (norm > 0) {
                    this.weights[j] = this.softThreshold(residual / n, this.alpha) / (norm / n);
                }
            }
            
            const diff = weightsOld.reduce((sum, w, i) => 
                sum + Math.abs(w - this.weights[i]), 0
            );
            if (diff < this.tol) break;
        }
    }

    predict(X) {
        return X.map(row => {
            let sum = this.bias;
            for (let i = 0; i < row.length; i++) {
                sum += row[i] * this.weights[i];
            }
            return sum;
        });
    }
}

class DecisionTreeRegressor {
    constructor(maxDepth = 5, minSamplesSplit = 2) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.tree = null;
    }

    calculateMSE(y) {
        const mean = y.reduce((a, b) => a + b, 0) / y.length;
        return y.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / y.length;
    }

    findBestSplit(X, y) {
        let bestMSE = Infinity;
        let bestFeature = null;
        let bestThreshold = null;
        
        for (let feature = 0; feature < X[0].length; feature++) {
            const values = X.map(row => row[feature]);
            const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
            
            for (let i = 0; i < uniqueValues.length - 1; i++) {
                const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
                
                const leftIndices = [];
                const rightIndices = [];
                
                X.forEach((row, idx) => {
                    if (row[feature] <= threshold) {
                        leftIndices.push(idx);
                    } else {
                        rightIndices.push(idx);
                    }
                });
                
                if (leftIndices.length === 0 || rightIndices.length === 0) continue;
                
                const leftY = leftIndices.map(idx => y[idx]);
                const rightY = rightIndices.map(idx => y[idx]);
                
                const mse = (leftY.length * this.calculateMSE(leftY) + 
                            rightY.length * this.calculateMSE(rightY)) / y.length;
                
                if (mse < bestMSE) {
                    bestMSE = mse;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }
        
        return { feature: bestFeature, threshold: bestThreshold };
    }

    buildTree(X, y, depth = 0) {
        if (depth >= this.maxDepth || y.length < this.minSamplesSplit || 
            new Set(y).size === 1) {
            return { value: y.reduce((a, b) => a + b, 0) / y.length };
        }
        
        const { feature, threshold } = this.findBestSplit(X, y);
        
        if (feature === null) {
            return { value: y.reduce((a, b) => a + b, 0) / y.length };
        }
        
        const leftIndices = [];
        const rightIndices = [];
        
        X.forEach((row, idx) => {
            if (row[feature] <= threshold) {
                leftIndices.push(idx);
            } else {
                rightIndices.push(idx);
            }
        });
        
        return {
            feature,
            threshold,
            left: this.buildTree(
                leftIndices.map(idx => X[idx]),
                leftIndices.map(idx => y[idx]),
                depth + 1
            ),
            right: this.buildTree(
                rightIndices.map(idx => X[idx]),
                rightIndices.map(idx => y[idx]),
                depth + 1
            )
        };
    }

    fit(X, y) {
        this.tree = this.buildTree(X, y);
    }

    predictSingle(x, node = this.tree) {
        if (node.value !== undefined) {
            return node.value;
        }
        
        if (x[node.feature] <= node.threshold) {
            return this.predictSingle(x, node.left);
        } else {
            return this.predictSingle(x, node.right);
        }
    }

    predict(X) {
        return X.map(x => this.predictSingle(x));
    }
}

class ModelEvaluator {
    static mse(yTrue, yPred) {
        const n = yTrue.length;
        let sum = 0;
        for (let i = 0; i < n; i++) {
            sum += Math.pow(yTrue[i] - yPred[i], 2);
        }
        return sum / n;
    }

    static rmse(yTrue, yPred) {
        return Math.sqrt(this.mse(yTrue, yPred));
    }

    static r2Score(yTrue, yPred) {
        const mean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
        const ssTot = yTrue.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);
        const ssRes = yTrue.reduce((sum, val, i) => sum + Math.pow(val - yPred[i], 2), 0);
        return 1 - (ssRes / ssTot);
    }
}
