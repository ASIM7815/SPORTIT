# IPL Sports Prediction System

A real-time IPL match prediction application with custom-built regression models implemented from scratch using pure JavaScript (NumPy equivalent).

## Features

- **Real-time Match Predictions**: Input live match data and get instant predictions
- **Win/Loss Probability**: Calculate winning and losing probabilities based on current match state
- **Score Predictions**: Predict final scores and required run rates
- **Custom ML Models**: Four regression models built from scratch:
  - Linear Regression
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)
  - Decision Tree Regressor
- **Model Evaluation**: Compare models using MSE, RMSE, and R² metrics
- **Beautiful UI/UX**: Modern, responsive design with smooth animations

## How to Use

1. Open `index.html` in your web browser
2. Select batting and bowling teams
3. Enter current match data:
   - Current runs
   - Wickets lost
   - Overs completed
   - Target score (if chasing)
4. Click "Predict Match Outcome"
5. View predictions including:
   - Win/Loss probabilities
   - Predicted final score
   - Current and required run rates
   - Model performance metrics

## Technical Implementation

### Custom Regression Models

All models are implemented from scratch without using any ML libraries:

- **Matrix Operations**: Custom matrix multiplication, transpose, and inverse operations
- **Linear Regression**: Ordinary Least Squares using normal equation
- **Ridge Regression**: L2 regularization to prevent overfitting
- **Lasso Regression**: L1 regularization with coordinate descent optimization
- **Decision Tree**: Recursive tree building with MSE-based splitting

### Feature Engineering

The system uses 10 key features:
- Current runs
- Wickets lost
- Overs completed
- Target score
- Current run rate
- Wickets remaining
- Overs remaining
- Powerplay indicator
- Death overs indicator
- Required run rate

### Model Training

- Generates synthetic training data based on IPL match patterns
- 80/20 train-test split
- Feature normalization using z-score standardization
- Automatic best model selection based on R² score

## Files Structure

```
├── index.html           # Main HTML structure
├── style.css           # Styling and animations
├── regression_models.js # Custom ML models implementation
├── app.js              # Application logic and predictions
└── README.md           # Documentation
```

## Browser Compatibility

Works on all modern browsers:
- Chrome
- Firefox
- Safari
- Edge

## Future Enhancements

- Historical match data integration
- Player-specific predictions
- Venue and pitch condition factors
- Real-time API integration
- Model persistence and retraining

## License

MIT License
