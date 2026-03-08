import joblib

model = joblib.load("model/final_model.pkl")

print("Model type:", type(model))

print("\nPipeline steps:")
print(model.named_steps)

lr = model.named_steps["model"]

print("\nModel coefficients:")
print(lr.coef_)

print("\nIntercept:")
print(lr.intercept_)