import difflib  # For suggestion logic

# Hardcoded stock prices
stock_prices = {
    "SENSEX": 85000,
    "NIFTY50": 25800,
    "NIFTYBANK": 59500,
    "RELIANCE": 1800,
    "BAL": 2350  # Bharti Airtel Limited
}

portfolio = {}
total_investment = 0

print("📈 Welcome to the Indian Stock Portfolio Tracker!")
print("Available stocks:", ", ".join(stock_prices.keys()))
print("Type 'done' when finished.\n")

# Step 1: Input stock name and quantity
while True:
    stock = input("Enter stock name (e.g., RELIANCE,BAL): ").upper()
    if stock == "DONE":
        break
    if stock not in stock_prices:
        print("⚠️ Stock not found.")
        suggestion = difflib.get_close_matches(stock, stock_prices.keys(), n=1)
        if suggestion:
            print(f"Did you mean: {suggestion[0]}?")
        continue

    try:
        quantity = int(input(f"Enter quantity for {stock}: "))
        if quantity < 0:
            print("Quantity must be non-negative.")
            continue
    except ValueError:
        print("Please enter a valid number.")
        continue

    portfolio[stock] = portfolio.get(stock, 0) + quantity
    investment = stock_prices[stock] * quantity
    total_investment += investment
    print(f"✅ Added {quantity} units of {stock} worth ₹{investment}\n")

# Step 2: Show total investment with percentage
print("\n📊 --- Portfolio Summary ---")
for stock, qty in portfolio.items():
    price = stock_prices[stock]
    value = price * qty
    percent = (value / total_investment) * 100
    print(f"{stock}: {qty} units x ₹{price} = ₹{value} ({percent:.2f}%)")

print(f"\n💰 Total Investment Value: ₹{total_investment}")

# Step 3: Optional - Save to file
save = input("\nDo you want to save the portfolio to a file? (yes/no): ").lower()
if save == "yes":
    file_type = input("Save as (txt/csv): ").lower()

    if file_type == "txt":
        with open("portfolio.txt", "w") as f:
            f.write("Stock Portfolio Summary:\n")
            for stock, qty in portfolio.items():
                value = stock_prices[stock] * qty
                percent = (value / total_investment) * 100
                f.write(f"{stock}: {qty} units x ₹{stock_prices[stock]} = ₹{value} ({percent:.2f}%)\n")
            f.write(f"\nTotal Investment: ₹{total_investment}\n")
        print("📁 Portfolio saved to portfolio.txt")

    elif file_type == "csv":
        with open("portfolio.csv", "w") as f:
            f.write("Stock,Quantity,Price,Total Value,Percentage\n")
            for stock, qty in portfolio.items():
                price = stock_prices[stock]
                value = qty * price
                percent = (value / total_investment) * 100
                f.write(f"{stock},{qty},{price},{value},{percent:.2f}%\n")
            f.write(f",,,Total Investment,₹{total_investment}\n")
        print("📁 Portfolio saved to portfolio.csv")
