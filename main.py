
import joblib

vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("logistic_regression_model.pkl")

# testing area
while True:
    lines = []
    continue_or_terminate = input("\nContinue (c) or terminate (t): ")
    if continue_or_terminate == "t":
        print("Manually terminated...")
        break
    elif continue_or_terminate == "c":
        print("Input sentences (for TERMINATION: (1) press ENTER, (2) insert ### and step (1) again):")
        pass
    elif continue_or_terminate == "":
        print("Input missing...")
        continue
    else:
        print("Wrong input...")
        continue
    while True:
        user_input = input("")
        lines.append(user_input)
        if user_input == "###":
            break

    lines = list(filter(None, lines))
    lines.remove("###")

    user_input_vec = vectorizer.transform(lines)
    prediction = model.predict(user_input_vec)

    # optional: showing probability
    probability = model.predict_proba(user_input_vec)
    print(f"  →{prediction[0]} (Confidence:{max(probability[0]):.2%})")
