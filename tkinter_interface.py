import tkinter as tk
from tkinter import ttk
from MLMODEL import GDPModel

# Create Tkinter window
window = tk.Tk()
window.title("GDP Prediction Interface")


# Function to get prediction based on user input
def predict_gdp():
    # Get user input for features
    total_indebtedness = float(entry_total_indebtedness.get())
    investment_rate = float(entry_investment_rate.get())
    jobs_creation = float(entry_jobs_creation.get())
    trade_deficit = float(entry_trade_deficit.get())

    # Make prediction using the model
    prediction = GDPModel().predict([total_indebtedness, investment_rate, jobs_creation, trade_deficit])

    # Display the prediction in the result label
    result_label.config(text=f"Predicted GDP: {prediction} million dinars")

# Labels
label_total_indebtedness = ttk.Label(window, text="Total Indebtedness:")
label_investment_rate = ttk.Label(window, text="Investment Rate:")
label_jobs_creation = ttk.Label(window, text="Jobs Creation:")
label_trade_deficit = ttk.Label(window, text="Trade Deficit:")

# Entry widgets for user input
entry_total_indebtedness = ttk.Entry(window)
entry_investment_rate = ttk.Entry(window)
entry_jobs_creation = ttk.Entry(window)
entry_trade_deficit = ttk.Entry(window)

# Button to trigger prediction
predict_button = ttk.Button(window, text="Predict GDP", command=predict_gdp)

# Result label
result_label = ttk.Label(window, text="Predicted GDP: ")

# Arrange widgets using grid layout
label_total_indebtedness.grid(row=0, column=0, padx=10, pady=5, sticky="w")
label_investment_rate.grid(row=1, column=0, padx=10, pady=5, sticky="w")
label_jobs_creation.grid(row=2, column=0, padx=10, pady=5, sticky="w")
label_trade_deficit.grid(row=3, column=0, padx=10, pady=5, sticky="w")

entry_total_indebtedness.grid(row=0, column=1, padx=10, pady=5)
entry_investment_rate.grid(row=1, column=1, padx=10, pady=5)
entry_jobs_creation.grid(row=2, column=1, padx=10, pady=5)
entry_trade_deficit.grid(row=3, column=1, padx=10, pady=5)

predict_button.grid(row=4, column=0, columnspan=2, pady=10)

result_label.grid(row=5, column=0, columnspan=2, pady=5)

# Start the Tkinter event loop
window.mainloop()
