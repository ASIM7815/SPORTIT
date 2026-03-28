// IPL Match Prediction Application

class IPLPredictor {
    constructor() {
        this.models = {
            linear: new LinearRegression(),
            ridge: new RidgeRegression(0.5),
            lasso: new LassoRegression(0.1),
            tree: new DecisionTreeRegressor(5, 2)
        };
        this.bestModel = null;
        this.modelMetrics = {};
        this.scaler = { mean: [], std: [] };
        
        this.initializeModels();
        this.setupEventListeners();
    }

    generateTrainingData() {
        const data = [];
        
        for (let i = 0; i < 500; i++) {
            const overs = Math.random() * 20;
            const wickets = Math.floor(Math.random() * 11);
            const currentRuns = Math.floor(Math.random() * 250);
            const target = Math.random() > 0.5 ? Math.floor(Math.random() * 250) : 0;
            
            const currentRR = overs > 0 ? currentRuns / overs : 0;
            const wicketsRemaining = 10 - wickets;
            const oversRemaining = 20 - overs;
            const isPowerplay = overs <= 6 ? 1 : 0;
            const isDeathOvers = overs >= 16 ? 1 : 0;
            const requiredRR = target > 0 && oversRemaining > 0 ? 
                (target - currentRuns) / oversRemaining : 0;
            
            const projectedScore = currentRuns + (oversRemaining * currentRR * 
                (wicketsRemaining / 10) * (1 + Math.random() * 0.3));
            
            const winProb = target > 0 ? 
                Math.max(0, Math.min(100, 50 + (currentRuns - target * (overs / 20)) / 2)) :
                Math.min(100, 50 + (currentRuns / 2));
            
            data.push({
                features: [currentRuns, wickets, overs, target, currentRR, 
                          wicketsRemaining, oversRemaining, isPowerplay, 
                          isDeathOvers, requiredRR],
                score: projectedScore,
                winProb: winProb
            });
        }
        
        return data;
    }

    normalizeData(data) {
        const features = data.map(d => d.features);
        const n = features[0].length;
        
        this.scaler.mean = [];
        this.scaler.std = [];
        
        for (let j = 0; j < n; j++) {
            const column = features.map(row => row[j]);
            const mean = column.reduce((a, b) => a + b, 0) / column.length;
            const variance = column.reduce((sum, val) => 
                sum + Math.pow(val - mean, 2), 0) / column.length;
            const std = Math.sqrt(variance) || 1;
            
            this.scaler.mean.push(mean);
            this.scaler.std.push(std);
        }
        
        return features.map(row => 
            row.map((val, j) => (val - this.scaler.mean[j]) / this.scaler.std[j])
        );
    }

    normalizeInput(features) {
        return features.map((val, j) => 
            (val - this.scaler.mean[j]) / this.scaler.std[j]
        );
    }

    initializeModels() {
        console.log('Training models...');
        
        const trainingData = this.generateTrainingData();
        const trainSize = Math.floor(trainingData.length * 0.8);
        
        const trainData = trainingData.slice(0, trainSize);
        const testData = trainingData.slice(trainSize);
        
        const X_train = this.normalizeData(trainData);
        const y_train = trainData.map(d => d.score);
        
        const X_test = testData.map(d => this.normalizeInput(d.features));
        const y_test = testData.map(d => d.score);
        
        let bestR2 = -Infinity;
        
        for (const [name, model] of Object.entries(this.models)) {
            model.fit(X_train, y_train);
            
            const predictions = model.predict(X_test);
            const mse = ModelEvaluator.mse(y_test, predictions);
            const rmse = ModelEvaluator.rmse(y_test, predictions);
            const r2 = ModelEvaluator.r2Score(y_test, predictions);
            
            this.modelMetrics[name] = { mse, rmse, r2 };
            
            if (r2 > bestR2) {
                bestR2 = r2;
                this.bestModel = name;
            }
            
            console.log(`${name}: R²=${r2.toFixed(4)}, RMSE=${rmse.toFixed(2)}`);
        }
        
        console.log(`Best model: ${this.bestModel}`);
    }

    predict(inputData) {
        const { currentRuns, wickets, overs, target } = inputData;
        
        const currentRR = overs > 0 ? currentRuns / overs : 0;
        const wicketsRemaining = 10 - wickets;
        const oversRemaining = 20 - overs;
        const isPowerplay = overs <= 6 ? 1 : 0;
        const isDeathOvers = overs >= 16 ? 1 : 0;
        const requiredRR = target > 0 && oversRemaining > 0 ? 
            (target - currentRuns) / oversRemaining : 0;
        
        const features = [
            currentRuns, wickets, overs, target || 0, currentRR,
            wicketsRemaining, oversRemaining, isPowerplay,
            isDeathOvers, requiredRR
        ];
        
        const normalizedFeatures = this.normalizeInput(features);
        
        const model = this.models[this.bestModel];
        const predictions = model.predict([normalizedFeatures]);
        const predictedScore = Math.max(currentRuns, Math.round(predictions[0]));
        
        let winProb, loseProb;
        
        if (target > 0) {
            const runsNeeded = target - currentRuns;
            const ballsRemaining = oversRemaining * 6;
            
            if (runsNeeded <= 0) {
                winProb = 100;
            } else if (wicketsRemaining === 0) {
                winProb = 0;
            } else if (oversRemaining <= 0) {
                winProb = currentRuns >= target ? 100 : 0;
            } else {
                const rrDiff = currentRR - requiredRR;
                const wicketFactor = wicketsRemaining / 10;
                const oversFactor = oversRemaining / 20;
                
                winProb = 50 + (rrDiff * 10) + (wicketFactor * 20) - (requiredRR * 2);
                winProb = Math.max(5, Math.min(95, winProb));
            }
        } else {
            const scoreFactor = (predictedScore / 200) * 50;
            const wicketFactor = (wicketsRemaining / 10) * 30;
            const oversFactor = (oversRemaining / 20) * 20;
            
            winProb = scoreFactor + wicketFactor + oversFactor;
            winProb = Math.max(10, Math.min(90, winProb));
        }
        
        loseProb = 100 - winProb;
        
        return {
            predictedScore,
            winProb: winProb.toFixed(1),
            loseProb: loseProb.toFixed(1),
            currentRR: currentRR.toFixed(2),
            requiredRR: requiredRR > 0 ? requiredRR.toFixed(2) : '-',
            runsNeeded: target > 0 ? Math.max(0, target - currentRuns) : '-',
            bestModel: this.bestModel,
            r2Score: this.modelMetrics[this.bestModel].r2.toFixed(4),
            rmse: this.modelMetrics[this.bestModel].rmse.toFixed(2)
        };
    }

    setupEventListeners() {
        const battingTeamSelect = document.getElementById('battingTeam');
        const bowlingTeamSelect = document.getElementById('bowlingTeam');
        
        // Function to update available options
        const updateTeamOptions = () => {
            const selectedBatting = battingTeamSelect.value;
            const selectedBowling = bowlingTeamSelect.value;
            
            // Update bowling team options
            Array.from(bowlingTeamSelect.options).forEach(option => {
                if (option.value === '') return; // Skip the "Select Team" option
                option.disabled = option.value === selectedBatting;
            });
            
            // Update batting team options
            Array.from(battingTeamSelect.options).forEach(option => {
                if (option.value === '') return; // Skip the "Select Team" option
                option.disabled = option.value === selectedBowling;
            });
        };
        
        // Listen for changes on both dropdowns
        battingTeamSelect.addEventListener('change', () => {
            updateTeamOptions();
        });
        
        bowlingTeamSelect.addEventListener('change', () => {
            updateTeamOptions();
        });
        
        document.getElementById('predictBtn').addEventListener('click', () => {
            this.handlePrediction();
        });
        
        document.querySelectorAll('.input-field').forEach(input => {
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handlePrediction();
                }
            });
        });
    }

    handlePrediction() {
        const battingTeam = document.getElementById('battingTeam').value;
        const bowlingTeam = document.getElementById('bowlingTeam').value;
        const currentRuns = parseFloat(document.getElementById('currentRuns').value) || 0;
        const wickets = parseInt(document.getElementById('wickets').value) || 0;
        const overs = parseFloat(document.getElementById('overs').value) || 0;
        const target = parseFloat(document.getElementById('target').value) || 0;
        
        if (!battingTeam || !bowlingTeam) {
            alert('Please select both teams');
            return;
        }
        
        if (battingTeam === bowlingTeam) {
            alert('Batting and bowling teams must be different');
            return;
        }
        
        if (wickets > 10 || overs > 20) {
            alert('Invalid input: Wickets cannot exceed 10 and overs cannot exceed 20');
            return;
        }
        
        const results = this.predict({
            currentRuns,
            wickets,
            overs,
            target
        });
        
        this.displayResults(results);
    }

    displayResults(results) {
        const resultsSection = document.getElementById('resultsSection');
        const backdrop = document.getElementById('backdrop');
        
        resultsSection.style.display = 'block';
        backdrop.style.display = 'block';
        
        this.currentResults = results;
        
        setTimeout(() => {
            document.getElementById('winProb').textContent = `${results.winProb}%`;
            document.getElementById('loseProb').textContent = `${results.loseProb}%`;
            document.getElementById('winProgress').style.width = `${results.winProb}%`;
            document.getElementById('loseProgress').style.width = `${results.loseProb}%`;
            
            document.getElementById('finalScore').textContent = results.predictedScore;
            document.getElementById('currentRR').textContent = results.currentRR;
            document.getElementById('requiredRR').textContent = results.requiredRR;
            document.getElementById('runsNeeded').textContent = results.runsNeeded;
            
            document.getElementById('bestModel').textContent = 
                results.bestModel.charAt(0).toUpperCase() + results.bestModel.slice(1);
            document.getElementById('r2Score').textContent = results.r2Score;
            document.getElementById('rmse').textContent = results.rmse;
        }, 100);
    }
}

class AnalyticsVisualizer {
    constructor() {
        this.charts = {};
    }

    createPieChart(winProb, loseProb) {
        const ctx = document.getElementById('pieChart').getContext('2d');
        
        if (this.charts.pie) this.charts.pie.destroy();
        
        this.charts.pie = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Win Probability', 'Loss Probability'],
                datasets: [{
                    data: [parseFloat(winProb), parseFloat(loseProb)],
                    backgroundColor: [
                        'rgba(0, 200, 83, 0.8)',
                        'rgba(255, 23, 68, 0.8)'
                    ],
                    borderColor: [
                        'rgba(0, 200, 83, 1)',
                        'rgba(255, 23, 68, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { size: 14 },
                            padding: 15
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    createBarChart(modelMetrics) {
        const ctx = document.getElementById('barChart').getContext('2d');
        
        if (this.charts.bar) this.charts.bar.destroy();
        
        const models = Object.keys(modelMetrics);
        const r2Scores = models.map(m => modelMetrics[m].r2);
        
        this.charts.bar = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: models.map(m => m.charAt(0).toUpperCase() + m.slice(1)),
                datasets: [{
                    label: 'R² Score',
                    data: r2Scores,
                    backgroundColor: [
                        'rgba(102, 126, 234, 0.8)',
                        'rgba(118, 75, 162, 0.8)',
                        'rgba(255, 152, 0, 0.8)',
                        'rgba(0, 200, 83, 0.8)'
                    ],
                    borderColor: [
                        'rgba(102, 126, 234, 1)',
                        'rgba(118, 75, 162, 1)',
                        'rgba(255, 152, 0, 1)',
                        'rgba(0, 200, 83, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            font: { size: 12 }
                        }
                    },
                    x: {
                        ticks: {
                            font: { size: 12 }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    createLineChart(currentRuns, overs, predictedScore) {
        const ctx = document.getElementById('lineChart').getContext('2d');
        
        if (this.charts.line) this.charts.line.destroy();
        
        const currentRR = overs > 0 ? currentRuns / overs : 0;
        const remainingOvers = 20 - overs;
        const projectedRuns = [];
        const oversData = [];
        
        for (let i = 0; i <= Math.ceil(overs); i++) {
            oversData.push(i);
            projectedRuns.push(Math.min(currentRuns, i * currentRR));
        }
        
        for (let i = Math.ceil(overs) + 1; i <= 20; i++) {
            oversData.push(i);
            const progress = (i - overs) / remainingOvers;
            projectedRuns.push(currentRuns + (predictedScore - currentRuns) * progress);
        }
        
        this.charts.line = new Chart(ctx, {
            type: 'line',
            data: {
                labels: oversData,
                datasets: [{
                    label: 'Projected Score',
                    data: projectedRuns,
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            font: { size: 12 }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Overs',
                            font: { size: 14 }
                        },
                        ticks: {
                            font: { size: 12 }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom'
                    }
                }
            }
        });
    }

    createRadarChart(wickets, overs, currentRR, requiredRR) {
        const ctx = document.getElementById('radarChart').getContext('2d');
        
        if (this.charts.radar) this.charts.radar.destroy();
        
        const wicketsRemaining = ((10 - wickets) / 10) * 100;
        const oversRemaining = ((20 - overs) / 20) * 100;
        const runRateScore = Math.min(100, (currentRR / 12) * 100);
        const rrDiff = requiredRR !== '-' ? 
            Math.max(0, 100 - (Math.abs(currentRR - parseFloat(requiredRR)) * 10)) : 50;
        const momentum = (wicketsRemaining + oversRemaining + runRateScore) / 3;
        
        this.charts.radar = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Wickets', 'Overs Left', 'Run Rate', 'RR Balance', 'Momentum'],
                datasets: [{
                    label: 'Match Factors',
                    data: [wicketsRemaining, oversRemaining, runRateScore, rrDiff, momentum],
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(102, 126, 234, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            font: { size: 10 }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    showAnalytics(results, inputData, modelMetrics) {
        document.getElementById('analyticsModal').style.display = 'block';
        
        setTimeout(() => {
            this.createPieChart(results.winProb, results.loseProb);
            this.createBarChart(modelMetrics);
            this.createLineChart(
                inputData.currentRuns, 
                inputData.overs, 
                results.predictedScore
            );
            this.createRadarChart(
                inputData.wickets,
                inputData.overs,
                parseFloat(results.currentRR),
                results.requiredRR
            );
        }, 100);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const predictor = new IPLPredictor();
    const visualizer = new AnalyticsVisualizer();
    console.log('IPL Predictor initialized successfully!');
    
    let lastInputData = null;
    
    const originalHandlePrediction = predictor.handlePrediction.bind(predictor);
    predictor.handlePrediction = function() {
        lastInputData = {
            currentRuns: parseFloat(document.getElementById('currentRuns').value) || 0,
            wickets: parseInt(document.getElementById('wickets').value) || 0,
            overs: parseFloat(document.getElementById('overs').value) || 0,
            target: parseFloat(document.getElementById('target').value) || 0
        };
        originalHandlePrediction();
    };
    
    document.getElementById('closeResults').addEventListener('click', () => {
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('backdrop').style.display = 'none';
    });
    
    document.getElementById('backdrop').addEventListener('click', () => {
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('analyticsModal').style.display = 'none';
        document.getElementById('backdrop').style.display = 'none';
    });
    
    document.getElementById('analyticsBtn').addEventListener('click', () => {
        if (predictor.currentResults && lastInputData) {
            visualizer.showAnalytics(
                predictor.currentResults, 
                lastInputData,
                predictor.modelMetrics
            );
        }
    });
    
    document.getElementById('closeAnalytics').addEventListener('click', () => {
        document.getElementById('analyticsModal').style.display = 'none';
    });
});
