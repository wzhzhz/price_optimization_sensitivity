import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# simulate a dataset with 10000 rows
def simulate_price_sensitivity_dataset(
    n_rows: int = 10_000,
    seed: int = 42,
    out_path: str = "price_sensitivity_dataset.csv",
):
    rng = np.random.default_rng(seed)

    
    # Categorical columns : flight_type
    flight_type = rng.choice(
        ["domestic", "international"],
        size=n_rows,
        p=[0.68, 0.32]    # set probabilities so domestic is more common than international
    )


    # Categorical columns : cabin_class
    cabin_class = rng.choice(
        ["economy", "premium_economy", "business"],
        size=n_rows,
        p=[0.78, 0.15, 0.07]  # probability of each calss
    )

  
    # binary variable : loyalty_program_member, whether customer has a membership
    affluence = rng.normal(0, 1, size=n_rows)  # randomly generate purchasing power
    deal_seeker = rng.normal(0, 1, size=n_rows)
    loyalty_latent = 0.25 * affluence - 0.10 * deal_seeker + rng.normal(0, 0.8, size=n_rows)
    loyalty_prob = sigmoid(-0.3 + 0.9 * loyalty_latent)
    loyalty_program_member = rng.binomial(1, loyalty_prob, size=n_rows).astype(int)

    # assign high purchasing power customers into premium economy or business cabin
    # assign low purchasing power customers into economy cabin
    nudge_mask = rng.random(n_rows) < 0.18
    for i in np.where(nudge_mask)[0]:
        if flight_type[i] == "international" and affluence[i] > 0.8:
            cabin_class[i] = rng.choice(["premium_economy", "business"], p=[0.55, 0.45])
        elif affluence[i] < -0.8:
            cabin_class[i] = "economy"

    # randomly generate average purchase frequency
    avg_purchase_frequency = np.clip(
        rng.lognormal(mean=0.35 + 0.12 * affluence, sigma=0.55, size=n_rows),
        0.2,
        20.0
    )

    # average order value
    cabin_multiplier = np.select(
        [cabin_class == "economy", cabin_class == "premium_economy", cabin_class == "business"],
        [1.0, 1.35, 2.4],
        default=1.0
    )
    flight_multiplier = np.where(flight_type == "international", 1.45, 1.0)
    base_aov = rng.lognormal(mean=4.6, sigma=0.45, size=n_rows)  
    avg_order_value = base_aov * cabin_multiplier * flight_multiplier
    avg_order_value = np.clip(avg_order_value, 30, 5000)

    # total spend
    horizon_factor = rng.normal(loc=1.8, scale=0.35, size=n_rows)  
    total_spent = avg_order_value * avg_purchase_frequency * np.clip(horizon_factor, 0.6, 3.5)
    total_spent *= rng.normal(loc=1.0, scale=0.15, size=n_rows)
    total_spent = np.clip(total_spent, 0, 200_000)


    # number of days since last purchase
    days_since_last_purchase = (
        rng.gamma(shape=2.2, scale=22, size=n_rows)
        + 18 * (1 - loyalty_program_member)
        + 6 * np.clip(deal_seeker, -2, 2)
    )
    days_since_last_purchase = np.clip(days_since_last_purchase, 0, 365).round().astype(int)

   
    # discount behavior
    # lower for high purchasing power customers, so negatively correlated with affluence
    # non-membership customers are more reply on discount, so higher for non-members
    discount_behavior = sigmoid(0.2 + 0.9 * deal_seeker - 0.35 * affluence + 0.25 * (1 - loyalty_program_member))
    discount_behavior = np.clip(discount_behavior + rng.normal(0, 0.05, size=n_rows), 0, 1)

    # number of days booked in advance
    days_in_advance = rng.gamma(shape=2.0, scale=12.0, size=n_rows) + np.where(flight_type == "international", 18, 6)
    days_in_advance = np.clip(days_in_advance, 0, 365).round().astype(int)

 
    cabin_effect = np.select(
        [cabin_class == "economy", cabin_class == "premium_economy", cabin_class == "business"],
        [0.0, 0.28, 0.70],
        default=0.0
    )
    flight_effect = np.where(flight_type == "international", 0.12, 0.0)

   
    log_spent = np.log1p(total_spent)
    log_aov = np.log1p(avg_order_value)

   
    increase_intensity = np.clip(rng.normal(0.12, 0.04, size=n_rows), 0.03, 0.25)

    # Probability model (logit)
    logit = (
        -0.35
        + 1.05 * loyalty_program_member
        - 1.25 * discount_behavior
        + 0.35 * np.log1p(avg_purchase_frequency)
        - 0.0045 * days_since_last_purchase
        + 0.10 * (log_spent - log_spent.mean())
        + 0.08 * (log_aov - log_aov.mean())
        + cabin_effect
        + flight_effect
        - 3.2 * increase_intensity * (0.6 + discount_behavior)  # price increase hurts more for discount seekers
        + rng.normal(0, 0.20, size=n_rows)  # unobserved factors
    )

    prob_continue = sigmoid(logit)

    # whether the customer buys after a price increase
    will_buy_after_price_increase = rng.binomial(1, prob_continue, size=n_rows).astype(int)

   
    df = pd.DataFrame({
        "customer_id": np.arange(1, n_rows + 1, dtype=np.int64),
        "total_spent": np.round(total_spent, 2),
        "avg_order_value": np.round(avg_order_value, 2),
        "avg_purchase_frequency": np.round(avg_purchase_frequency, 2),
        "days_since_last_purchase": days_since_last_purchase,
        "discount_behavior": np.round(discount_behavior, 2),
        "loyalty_program_member": loyalty_program_member,
        "days_in_advance": days_in_advance,
        "flight_type": flight_type,
        "cabin_class": cabin_class,
        "will_buy_after_price_increase": will_buy_after_price_increase
    })

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | shape={df.shape} | continuation_rate={df['will_buy_after_price_increase'].mean():.3f}")
    return df

if __name__ == "__main__":
    simulate_price_sensitivity_dataset(n_rows=10_000, seed=42, out_path="price_sensitivity_data.csv")
