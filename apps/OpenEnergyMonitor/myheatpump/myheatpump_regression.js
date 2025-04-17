/**
 * Performs linear regression on paired data.
 * y = mx + b
 * @param {number[]} x - Array of x values (independent variable, e.g., time in seconds).
 * @param {number[]} y - Array of y values (dependent variable, e.g., ln(deltaT/deltaT0)).
 * @returns {object|null} Object with 'slope' (m) and 'intercept' (b), or null if regression is not possible.
 */
function linearRegression(x, y) {
    const n = x.length;
    if (n < 2 || n !== y.length) {
        console.error("Linear regression requires at least 2 points and equal length arrays.");
        return null; // Not enough data or mismatched arrays
    }

    let sum_x = 0;
    let sum_y = 0;
    let sum_xy = 0;
    let sum_xx = 0;
    let sum_yy = 0; // Needed for R-squared, not strictly required for slope/intercept

    for (let i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
        sum_yy += y[i] * y[i];
    }

    const denominator = (n * sum_xx - sum_x * sum_x);
    if (Math.abs(denominator) < 1e-10) { // Avoid division by zero if all x are the same
         console.error("Linear regression failed: Denominator is zero (all x values are likely the same).");
         return null;
    }

    const slope = (n * sum_xy - sum_x * sum_y) / denominator;
    const intercept = (sum_y - slope * sum_x) / n;

    // Optional: Calculate R-squared (coefficient of determination)
    let ssr = 0;
    for (let i = 0; i < n; i++) {
        const fit = slope * x[i] + intercept;
        ssr += (fit - sum_y / n) ** 2;
    }
    const sst = sum_yy - (sum_y * sum_y) / n;
    const r2 = (sst === 0) ? 1 : ssr / sst; // Handle case where all y are the same

    return {
        slope: slope,
        intercept: intercept,
        r2: r2 // Uncomment if you want R-squared
    };
}



/**
 * Performs Multilinear Regression using matrix calculations via math.js.
 * Calculates coefficients, standard errors, t-stats, p-values, and confidence intervals.
 * Finds coefficients (β₁, β₂, ...) and intercept (β₀) for the model:
 * Y = β₀ + β₁X₁ + β₂X₂ + ... + β<0xE2><0x82><0x99>X<0xE2><0x82><0x99>
 * Calculates β = (XᵀX)⁻¹XᵀY
 *
 * @param {number[][]} independentVars - Array of arrays for independent variables (X₁, X₂, ...).
 * @param {number[]} dependentVar - Array for the dependent variable (Y).
 * @param {number} [confidenceLevel=0.95] - The desired confidence level for intervals (e.g., 0.95 for 95%).
 *
 * @returns {object|null} An object containing regression results including:
 *                        - `coefficients`: Array [β₁, β₂, ...].
 *                        - `intercept`: The intercept (β₀).
 *                        - `beta`: Full coefficient vector [β₀, β₁, β₂, ...].
 *                        - `r2`: R-squared value.
 *                        - `standardErrors`: Array of SEs for [β₀, β₁, β₂, ...].
 *                        - `tStats`: Array of t-statistics for [β₀, β₁, β₂, ...].
 *                        - `pValues`: Array of p-values for [β₀, β₁, β₂, ...].
 *                        - `confidenceIntervals`: Array of CIs [[lower, upper], ...] for [β₀, β₁, β₂, ...].
 *                        - `degreesOfFreedom`: Error degrees of freedom (n-p).
 *                        Or null if regression fails.
 */
function multilinearRegression(independentVars, dependentVar, confidenceLevel = 0.95) {
    // 1. Check if libraries are available
    if (typeof math === 'undefined') {
        console.error("Multilinear Regression Error: The 'math.js' library is not loaded.");
        return null;
    }
     if (typeof jStat === 'undefined') {
        console.error("Multilinear Regression Error: The 'jStat' library is not loaded (needed for confidence intervals).");
        // Optionally, proceed without CIs, or return null
         // return null;
         console.warn("  >> Proceeding without calculating confidence intervals, p-values, etc.");
    }


    // 2. Input Validation
    if (!independentVars || independentVars.length === 0 || !dependentVar) { /* ... */ return null; }
    const numIndependentVars = independentVars.length;
    const numDataPoints = dependentVar.length;
    const p = numIndependentVars + 1; // Number of parameters (coefficients + intercept)

    if (numDataPoints <= p) { // Need more points than parameters estimated
        console.error(`Multilinear Regression Error: Insufficient data points (${numDataPoints}). Need more than ${p} for ${numIndependentVars} independent vars + intercept.`);
        return null;
    }
    // Check array lengths consistency... (same as before)

    try {
        // 3. Construct Design Matrix X (with intercept column)
        const X_data = [];
        for (let i = 0; i < numDataPoints; i++) {
            const row = [1]; // Intercept column
            for (let j = 0; j < numIndependentVars; j++) {
                row.push(independentVars[j][i]);
            }
            X_data.push(row);
        }
        const X = math.matrix(X_data);

        // 4. Create Dependent Variable Vector Y
        const Y = math.matrix(dependentVar);

        // 5. Calculate Coefficients: β = (XᵀX)⁻¹XᵀY
        const XT = math.transpose(X);
        const XTX = math.multiply(XT, X);
        const XTX_inv = math.inv(XTX); // (XᵀX)⁻¹
        const XTY = math.multiply(XT, Y);
        const beta_vector = math.multiply(XTX_inv, XTY); // Result is a column matrix
        const beta = beta_vector.valueOf().flat(); // .valueOf().flat() converts math.js matrix to plain 1D array

        // beta array is [intercept, coeff_X1, coeff_X2, ...]
        const intercept = beta[0];
        const coefficients = beta.slice(1);

        // 6. Calculate Residuals and SSE
        const Y_predicted_vector = math.multiply(X, beta_vector);
        const residuals_vector = math.subtract(Y, Y_predicted_vector);
        const residuals = residuals_vector.valueOf().flat(); // Plain 1D array of residuals
        const ss_res = residuals.reduce((sum, r) => sum + r * r, 0); // Sum of Squared Errors (SSE)

        // 7. Calculate R-squared
        const Y_mean = math.mean(dependentVar);
        const ss_tot = dependentVar.reduce((sum, y) => sum + Math.pow(y - Y_mean, 2), 0);
        const r2 = (ss_tot === 0) ? 1 : 1 - (ss_res / ss_tot);

        // --- Statistical Inference Calculations ---
        let standardErrors = null, tStats = null, pValues = null, confidenceIntervals = null;
        const degreesOfFreedom = numDataPoints - p;

        if (degreesOfFreedom <= 0) {
            console.warn("Multilinear Regression Warning: Degrees of freedom is not positive. Cannot calculate standard errors or confidence intervals.");
        } else if (typeof jStat !== 'undefined') { // Only proceed if jStat is loaded and df > 0
            try {
                // Estimate variance of residuals: s² = SSE / (n - p)
                const residual_variance = ss_res / degreesOfFreedom;

                // Standard Errors: sqrt(diagonal elements of s² * (XᵀX)⁻¹)
                // math.diag extracts the diagonal from the matrix
                const diag_XTX_inv = math.diag(XTX_inv).valueOf(); // Get diagonal as plain array
                standardErrors = diag_XTX_inv.map(d => math.sqrt(residual_variance * d));

                // t-statistics: coefficient / standard_error
                tStats = beta.map((b, i) => standardErrors[i] === 0 ? NaN : b / standardErrors[i]); // Avoid division by zero

                // p-values (two-tailed test: H0: βj = 0)
                pValues = tStats.map(t => isNaN(t) ? NaN : jStat.studentt.cdf(-Math.abs(t), degreesOfFreedom) * 2);

                // Confidence Intervals: coeff ± t_crit * SE
                const alpha = 1 - confidenceLevel;
                const t_crit = jStat.studentt.inv(1 - alpha / 2, degreesOfFreedom); // Critical t-value

                confidenceIntervals = beta.map((b, i) => {
                    const marginOfError = t_crit * standardErrors[i];
                    return [b - marginOfError, b + marginOfError];
                });

             } catch (statError) {
                 console.error("Error during statistical inference calculations (SE, CI):", statError);
                 // Set inference results to null if calculation fails
                 standardErrors = null; tStats = null; pValues = null; confidenceIntervals = null;
             }
        } // End if jStat available and df > 0


        return {
            coefficients: coefficients, // [β₁, β₂, ...]
            intercept: intercept,       // β₀
            beta: beta,                 // Full vector [β₀, β₁, β₂, ...]
            r2: r2,
            standardErrors: standardErrors, // SE for [β₀, β₁, β₂, ...] or null
            tStats: tStats,             // t-stats for [β₀, β₁, β₂, ...] or null
            pValues: pValues,           // p-values for [β₀, β₁, β₂, ...] or null
            confidenceIntervals: confidenceIntervals, // CIs for [β₀, β₁, β₂, ...] or null
            degreesOfFreedom: degreesOfFreedom,
            sse: ss_res,
            n: numDataPoints,
            p: p
        };

    } catch (error) {
        console.error("Error during multilinear regression calculation using math.js:", error);
        if (error.message && error.message.includes("Cannot calculate inverse")) {
             console.error("  >> This often indicates perfect multicollinearity among independent variables.");
        }
        return null;
    }
}

/**
 * Performs Multilinear Regression using matrix calculations via math.js.
 * Calculates coefficients, standard errors, t-stats, p-values, and confidence intervals.
 * Solves the normal equations (XᵀX)β = XᵀY using LU decomposition (math.lusolve)
 * for better numerical stability compared to direct matrix inversion.
 * Finds coefficients (β₁, β₂, ...) and intercept (β₀) for the model:
 * Y = β₀ + β₁X₁ + β₂X₂ + ... + β<0xE2><0x82><0x99>X<0xE2><0x82><0x99>
 *
 * @param {number[][]} independentVars - Array of arrays for independent variables (X₁, X₂, ...).
 * Each inner array should have length n (number of data points).
 * @param {number[]} dependentVar - Array for the dependent variable (Y) with length n.
 * @param {number} [confidenceLevel=0.95] - The desired confidence level for intervals (e.g., 0.95 for 95%).
 *
 * @returns {object|null} An object containing regression results including:
 *                        - `coefficients`: Array [β₁, β₂, ...].
 *                        - `intercept`: The intercept (β₀).
 *                        - `beta`: Full coefficient vector [β₀, β₁, β₂, ...].
 *                        - `r2`: R-squared value.
 *                        - `standardErrors`: Array of SEs for [β₀, β₁, β₂, ...] or null.
 *                        - `tStats`: Array of t-statistics for [β₀, β₁, β₂, ...] or null.
 *                        - `pValues`: Array of p-values for [β₀, β₁, β₂, ...] or null.
 *                        - `confidenceIntervals`: Array of CIs [[lower, upper], ...] for [β₀, β₁, β₂, ...] or null.
 *                        - `degreesOfFreedom`: Error degrees of freedom (n-p).
 *                        - `sse`: Sum of Squared Errors.
 *                        - `n`: Number of data points.
 *                        - `p`: Number of parameters (coefficients + intercept).
 *                        Or null if regression fails.
 */
function multilinearRegressionStable(independentVars, dependentVar, confidenceLevel = 0.95) {
    // 1. Check if libraries are available
    if (typeof math === 'undefined') {
        console.error("Multilinear Regression Error: The 'math.js' library is not loaded.");
        return null;
    }
     if (typeof jStat === 'undefined') {
        console.warn("Multilinear Regression Warning: The 'jStat' library is not loaded. Confidence intervals, p-values, etc. will not be calculated.");
    }

    // 2. Input Validation
    if (!Array.isArray(independentVars) || independentVars.length === 0 || !Array.isArray(dependentVar)) {
         console.error("Multilinear Regression Error: Invalid input types. independentVars should be Array<Array<number>>, dependentVar should be Array<number>.");
         return null;
    }
    const numIndependentVars = independentVars.length;
    const numDataPoints = dependentVar.length;

     if (numDataPoints === 0) {
        console.error("Multilinear Regression Error: No data points provided.");
        return null;
    }
    if (numIndependentVars === 0) {
         console.error("Multilinear Regression Error: No independent variables provided.");
         return null;
    }

     // Check consistency of lengths within independentVars and with dependentVar
    for (let j = 0; j < numIndependentVars; j++) {
        if (!Array.isArray(independentVars[j])) {
             console.error(`Multilinear Regression Error: Independent variable at index ${j} is not an array.`);
             return null;
        }
        if (independentVars[j].length !== numDataPoints) {
            console.error(`Multilinear Regression Error: Length mismatch. Dependent variable has ${numDataPoints} points, but independent variable ${j} has ${independentVars[j].length} points.`);
            return null;
        }
    }

    const p = numIndependentVars + 1; // Number of parameters (coefficients + intercept)

    if (numDataPoints < p) { // Need at least as many points as parameters estimated
        console.error(`Multilinear Regression Error: Insufficient data points (${numDataPoints}). Need at least ${p} points for ${numIndependentVars} independent vars + intercept to avoid underdetermined system.`);
        return null;
    }
     if (numDataPoints === p) {
         console.warn(`Multilinear Regression Warning: Number of data points (${numDataPoints}) equals number of parameters (${p}). The fit will be exact (R²=1), but statistical inference (SE, p-values) may not be meaningful.`);
         // Allow calculation to proceed, but inference might yield NaNs or infinities later
     }


    try {
        // 3. Construct Design Matrix X (with intercept column)
        const X_data = [];
        for (let i = 0; i < numDataPoints; i++) {
            const row = [1]; // Intercept column
            for (let j = 0; j < numIndependentVars; j++) {
                 // Ensure data is numeric
                const val = Number(independentVars[j][i]);
                if (isNaN(val)) {
                    console.error(`Multilinear Regression Error: Non-numeric value found in independent variables at index [${j}][${i}].`);
                    return null;
                }
                row.push(val);
            }
            X_data.push(row);
        }
        const X = math.matrix(X_data); // size n x p

        // 4. Create Dependent Variable Vector Y
        const Y_data = dependentVar.map(y => {
             const val = Number(y);
             if (isNaN(val)) {
                console.error(`Multilinear Regression Error: Non-numeric value found in dependent variable.`);
                throw new Error("Non-numeric dependent variable"); // Throw to be caught by outer try-catch
             }
             return val;
        });
        const Y = math.matrix(Y_data); // size n x 1

        // 5. Calculate Intermediate Matrices: Xᵀ and XᵀX
        const XT = math.transpose(X); // size p x n
        const XTX = math.multiply(XT, X); // size p x p

        // 6. Solve Normal Equations for Beta: (XᵀX)β = XᵀY
        const XTY = math.multiply(XT, Y); // size p x 1
        // Use lusolve for stability instead of inv:
        const beta_vector = math.lusolve(XTX, XTY); // size p x 1
        const beta = beta_vector.valueOf().flat(); // Convert math.js matrix to plain 1D array

        // beta array is [intercept, coeff_X1, coeff_X2, ...]
        const intercept = beta[0];
        const coefficients = beta.slice(1);

        // 7. Calculate Residuals and SSE
        const Y_predicted_vector = math.multiply(X, beta_vector);
        const residuals_vector = math.subtract(Y, Y_predicted_vector);
        const residuals = residuals_vector.valueOf().flat(); // Plain 1D array
        const ss_res = residuals.reduce((sum, r) => sum + r * r, 0); // Sum of Squared Errors (SSE)

        // 8. Calculate R-squared
        const Y_mean = math.mean(Y_data); // Use validated Y_data
        const ss_tot = Y_data.reduce((sum, y) => sum + Math.pow(y - Y_mean, 2), 0);
        // Handle case where all Y values are the same (ss_tot = 0)
        const r2 = (ss_tot === 0) ? 1 : (ss_res === 0 && ss_tot === 0 ? 1 : Math.max(0, 1 - (ss_res / ss_tot))); // Ensure R2 is not negative due to floating point issues

        // --- Statistical Inference Calculations ---
        let standardErrors = null, tStats = null, pValues = null, confidenceIntervals = null;
        const degreesOfFreedom = numDataPoints - p;

        if (degreesOfFreedom <= 0) {
            console.warn("Multilinear Regression Warning: Degrees of freedom is not positive. Cannot calculate standard errors, p-values, or confidence intervals.");
        } else if (typeof jStat !== 'undefined') { // Only proceed if jStat is loaded and df > 0
            try {
                // Estimate variance of residuals: s² = SSE / (n - p)
                const residual_variance = ss_res / degreesOfFreedom;

                // Calculate diagonal elements of (XᵀX)⁻¹ without full inversion
                // We need diag( (XᵀX)⁻¹ ) to compute SEs.
                // Solve (XᵀX)cᵢ = eᵢ for cᵢ, where eᵢ is the i-th standard basis vector.
                // The i-th element of cᵢ is the i-th diagonal element of (XᵀX)⁻¹.
                const diag_XTX_inv = [];
                const identity_p = math.identity(p); // Create p x p identity matrix

                for (let i = 0; i < p; i++) {
                    // Extract the i-th column (standard basis vector eᵢ) from the identity matrix
                    const e_i = math.subset(identity_p, math.index(math.range(0, p), i));
                    // Solve the system (XᵀX)cᵢ = eᵢ
                    const c_i_vector = math.lusolve(XTX, e_i);
                    // The i-th diagonal element is the i-th element of the solution cᵢ
                    const diag_element = math.subset(c_i_vector, math.index(i, 0)); // Access element [i, 0] of column vector
                    diag_XTX_inv.push(diag_element);
                }

                // Standard Errors: sqrt( s² * diag((XᵀX)⁻¹) )
                standardErrors = diag_XTX_inv.map(d => {
                    const variance_beta_i = residual_variance * d;
                    // Handle potential negative variance due to numerical issues in edge cases
                    return variance_beta_i < 0 ? NaN : math.sqrt(variance_beta_i);
                 });

                // t-statistics: coefficient / standard_error
                tStats = beta.map((b, i) => (standardErrors[i] === 0 || isNaN(standardErrors[i])) ? NaN : b / standardErrors[i]); // Avoid division by zero or NaN SE

                // p-values (two-tailed test: H0: βj = 0)
                pValues = tStats.map(t => isNaN(t) ? NaN : jStat.studentt.cdf(-Math.abs(t), degreesOfFreedom) * 2);

                // Confidence Intervals: coeff ± t_crit * SE
                const alpha = 1 - confidenceLevel;
                // Handle potential issues with degreesOfFreedom <= 0 for jStat.inv
                let t_crit = NaN;
                 if (degreesOfFreedom > 0) {
                     try {
                        t_crit = jStat.studentt.inv(1 - alpha / 2, degreesOfFreedom);
                     } catch (jStatError){
                         console.error("Error calculating critical t-value with jStat:", jStatError);
                     }
                 }


                confidenceIntervals = beta.map((b, i) => {
                    if (isNaN(t_crit) || isNaN(standardErrors[i])) {
                        return [NaN, NaN];
                    }
                    const marginOfError = t_crit * standardErrors[i];
                    return [b - marginOfError, b + marginOfError];
                });

             } catch (statError) {
                 console.error("Error during statistical inference calculations (SE, CI, etc.):", statError);
                 // Set inference results to null if calculation fails
                 standardErrors = null; tStats = null; pValues = null; confidenceIntervals = null;
             }
        } // End if jStat available and df > 0

        return {
            coefficients: coefficients,
            intercept: intercept,
            beta: beta,
            r2: r2,
            standardErrors: standardErrors,
            tStats: tStats,
            pValues: pValues,
            confidenceIntervals: confidenceIntervals,
            degreesOfFreedom: degreesOfFreedom,
            sse: ss_res,
            n: numDataPoints,
            p: p
        };

    } catch (error) {
        console.error("Error during multilinear regression calculation using math.js:", error);
        // Check for specific math.js errors related to solving
        if (error.message && (error.message.includes("matrix is singular") || error.message.includes("Dimension mismatch"))) {
             console.error("  >> This may indicate perfect multicollinearity or other issues with the input data/matrix structure.");
        }
        return null;
    }
}

/**
 * Compares two regression result objects and prints the comparison to the console.
 *
 * @param {object|null} results1 - The first regression result object.
 * @param {object|null} results2 - The second regression result object.
 * @param {string} [label1='Result 1'] - Label for the first result set in the output.
 * @param {string} [label2='Result 2'] - Label for the second result set in the output.
 * @param {number} [tolerance=1e-8] - Tolerance for considering floating-point numbers "equal".
 * @param {number} [digits=8] - Number of decimal places to display for floating-point numbers.
 */
function compareRegressionResults(results1, results2, label1 = 'Result 1 (Inv)', label2 = 'Result 2 (Solve)', tolerance = 1e-8, digits = 8) {

    // --- Helper Functions ---

    /** Formats a number or returns 'N/A' if null/undefined/NaN */
    function formatNum(num, d = digits) {
        if (num === null || typeof num === 'undefined' || isNaN(num)) {
            return 'N/A';
        }
        return num.toFixed(d);
    }

    /** Compares two scalar values */
    function compareValues(name, val1, val2, l1, l2, fmt, tol) {
        const fVal1 = fmt(val1);
        const fVal2 = fmt(val2);
        let diffStr = '';
        if (fVal1 !== 'N/A' && fVal2 !== 'N/A') {
            const diff = Math.abs(val1 - val2);
            diffStr = ` | Diff: ${fmt(diff)}${diff > tol ? ' <-- SIGNIFICANT' : ''}`;
        } else if (fVal1 !== fVal2) {
             diffStr = ` | Difference in availability`;
        }
        console.log(`  ${name}:`);
        console.log(`    ${l1}: ${fVal1}`);
        console.log(`    ${l2}: ${fVal2}${diffStr}`);
    }

     /** Compares two arrays of numbers */
    function compareArrays(name, arr1, arr2, l1, l2, fmt, tol) {
        console.log(`  ${name}:`);
        const len1 = Array.isArray(arr1) ? arr1.length : 0;
        const len2 = Array.isArray(arr2) ? arr2.length : 0;

        if (!Array.isArray(arr1) && !Array.isArray(arr2)) {
             console.log(`    Both N/A`);
             return;
        }
         if (!Array.isArray(arr1)) {
             console.log(`    ${l1}: N/A`);
             console.log(`    ${l2}: [${arr2.map(v => fmt(v)).join(', ')}]`);
             return;
         }
         if (!Array.isArray(arr2)) {
             console.log(`    ${l1}: [${arr1.map(v => fmt(v)).join(', ')}]`);
             console.log(`    ${l2}: N/A`);
             return;
         }

        if (len1 !== len2) {
            console.log(`    ${l1}: Length ${len1} [${arr1.map(v => fmt(v)).join(', ')}]`);
            console.log(`    ${l2}: Length ${len2} [${arr2.map(v => fmt(v)).join(', ')}] <-- LENGTH MISMATCH`);
            return;
        }

        if (len1 === 0) {
             console.log(`    Both empty arrays`);
             return;
        }

        console.log(`    Index | ${l1.padEnd(digits + 4)} | ${l2.padEnd(digits + 4)} | Diff`);
        console.log(`    ------|-${'-'.repeat(digits + 4)}|-${'-'.repeat(digits + 4)}|-------`);

        let maxDiff = 0;
        arr1.forEach((val1, i) => {
            const val2 = arr2[i];
            const fVal1 = fmt(val1);
            const fVal2 = fmt(val2);
            let diffStr = 'N/A';
            let diff = NaN;
             if (fVal1 !== 'N/A' && fVal2 !== 'N/A') {
                diff = Math.abs(val1 - val2);
                maxDiff = Math.max(maxDiff, diff);
                diffStr = fmt(diff);
             }
            console.log(`    ${String(i).padEnd(5)} | ${fVal1.padEnd(digits + 4)} | ${fVal2.padEnd(digits + 4)} | ${diffStr}${diff > tol ? ' <--' : ''}`);
        });
         if (maxDiff > tol) {
             console.log(`    Max Difference: ${fmt(maxDiff)} <-- SIGNIFICANT`);
         } else {
             console.log(`    Max Difference: ${fmt(maxDiff)}`);
         }
    }

     /** Compares two arrays of confidence intervals ([lower, upper]) */
    function compareIntervals(name, intervals1, intervals2, l1, l2, fmt, tol) {
        console.log(`  ${name}:`);
        const len1 = Array.isArray(intervals1) ? intervals1.length : 0;
        const len2 = Array.isArray(intervals2) ? intervals2.length : 0;

         if (!Array.isArray(intervals1) && !Array.isArray(intervals2)) {
             console.log(`    Both N/A`);
             return;
        }
         if (!Array.isArray(intervals1)) {
             console.log(`    ${l1}: N/A`);
             console.log(`    ${l2}: ${len2} intervals`);
             return;
         }
         if (!Array.isArray(intervals2)) {
             console.log(`    ${l1}: ${len1} intervals`);
             console.log(`    ${l2}: N/A`);
             return;
         }

        if (len1 !== len2) {
            console.log(`    ${l1}: ${len1} intervals <-- LENGTH MISMATCH`);
            console.log(`    ${l2}: ${len2} intervals`);
            // Optionally print contents if lengths differ
            return;
        }
         if (len1 === 0) {
             console.log(`    Both empty arrays`);
             return;
         }

        console.log(`    Index | ${l1} Lower        | ${l2} Lower        | Lower Diff | ${l1} Upper        | ${l2} Upper        | Upper Diff`);
        console.log(`    ------|-------------------|-------------------|------------|-------------------|-------------------|------------`);

        let maxDiffOverall = 0;

        intervals1.forEach((int1, i) => {
            const int2 = intervals2[i];
            const [low1, up1] = Array.isArray(int1) ? int1 : [NaN, NaN];
            const [low2, up2] = Array.isArray(int2) ? int2 : [NaN, NaN];

            const fLow1 = fmt(low1);
            const fLow2 = fmt(low2);
            const fUp1 = fmt(up1);
            const fUp2 = fmt(up2);

            let diffLow = NaN, diffUp = NaN;
            let diffLowStr = 'N/A', diffUpStr = 'N/A';
            let lowSig = '', upSig = '';

            if (fLow1 !== 'N/A' && fLow2 !== 'N/A') {
                 diffLow = Math.abs(low1 - low2);
                 maxDiffOverall = Math.max(maxDiffOverall, diffLow);
                 diffLowStr = fmt(diffLow);
                 if (diffLow > tol) lowSig = '<--';
            }
             if (fUp1 !== 'N/A' && fUp2 !== 'N/A') {
                 diffUp = Math.abs(up1 - up2);
                 maxDiffOverall = Math.max(maxDiffOverall, diffUp);
                 diffUpStr = fmt(diffUp);
                  if (diffUp > tol) upSig = '<--';
            }

             console.log(`    ${String(i).padEnd(5)} | ${fLow1.padEnd(17)} | ${fLow2.padEnd(17)} | ${diffLowStr.padEnd(10)} ${lowSig.padEnd(3)}| ${fUp1.padEnd(17)} | ${fUp2.padEnd(17)} | ${diffUpStr.padEnd(10)} ${upSig}`);
        });
         if (maxDiffOverall > tol) {
             console.log(`    Max Difference in bounds: ${fmt(maxDiffOverall)} <-- SIGNIFICANT`);
         } else {
             console.log(`    Max Difference in bounds: ${fmt(maxDiffOverall)}`);
         }

    }

    // --- Main Comparison Logic ---

    console.group(`Comparing Regression Results: ${label1} vs ${label2}`);

    if (!results1 || typeof results1 !== 'object') {
        console.error(`${label1} is null or not an object.`);
        console.groupEnd();
        return;
    }
    if (!results2 || typeof results2 !== 'object') {
        console.error(`${label2} is null or not an object.`);
        console.groupEnd();
        return;
    }

    // --- Compare Metadata ---
    console.log("--- Metadata ---");
    compareValues('N (Data Points)', results1.n, results2.n, label1, label2, formatNum, 0); // Expect exact match
    compareValues('P (Parameters)', results1.p, results2.p, label1, label2, formatNum, 0); // Expect exact match
    compareValues('Degrees of Freedom', results1.degreesOfFreedom, results2.degreesOfFreedom, label1, label2, formatNum, 0); // Expect exact match

    // --- Compare Model Fit ---
    console.log("\n--- Model Fit ---");
    compareValues('R-squared', results1.r2, results2.r2, label1, label2, formatNum, tolerance);
    compareValues('SSE', results1.sse, results2.sse, label1, label2, formatNum, tolerance);

    // --- Compare Coefficients ---
    console.log("\n--- Coefficients (Beta Vector) ---");
    // Comparing the full 'beta' vector ([intercept, coeff1, coeff2, ...])
    compareArrays('Beta Vector', results1.beta, results2.beta, label1, label2, formatNum, tolerance);
    // Or compare intercept and coefficients separately if preferred:
    // compareValues('Intercept (β₀)', results1.intercept, results2.intercept, label1, label2, formatNum, tolerance);
    // compareArrays('Coefficients (β₁, β₂, ...)', results1.coefficients, results2.coefficients, label1, label2, formatNum, tolerance);


    // --- Compare Statistical Inference Results ---
    console.log("\n--- Statistical Inference ---");
    compareArrays('Standard Errors', results1.standardErrors, results2.standardErrors, label1, label2, formatNum, tolerance);
    compareArrays('T-Statistics', results1.tStats, results2.tStats, label1, label2, formatNum, tolerance);
    compareArrays('P-Values', results1.pValues, results2.pValues, label1, label2, formatNum, tolerance);
    compareIntervals('Confidence Intervals', results1.confidenceIntervals, results2.confidenceIntervals, label1, label2, formatNum, tolerance);


    console.groupEnd();
}
