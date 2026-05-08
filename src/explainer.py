import numpy as np
import shap

def get_shap_explainer(model):
    return shap.TreeExplainer(model)

def get_shap_values(explainer, input_array):
    shap_values = explainer.shap_values(input_array)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_values = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[0])

    base_price = float(np.expm1(base_value))
    return np.array(shap_values).flatten(), base_price


def _price_impact(shap_val, base_price):
    """Convert log-scale SHAP value to approximate rupee impact."""
    return abs(float(np.expm1(abs(float(shap_val)))) - 1) * base_price


def explain(shap_vals, feature_names, feature_vals, predicted_price, base_price):
    """
    Generate structured price explanations for both non-tech and tech users.

    Returns dict with:
      - summary:         one-line deal verdict
      - deal_score:      int 1-10
      - price_breakdown: list of dicts per feature
      - warnings:        list of important buyer/seller alerts
      - tech_table:      list of dicts for technical users
    """
    shap_vals = np.array(shap_vals).flatten()
    if len(shap_vals) != len(feature_names):
        shap_vals = np.resize(shap_vals, len(feature_names))

    vals    = dict(zip(feature_names, [float(v) for v in feature_vals]))
    impacts = dict(zip(feature_names, shap_vals))
    sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    price_breakdown = []
    warnings        = []
    tech_table      = []

    for feature, shap_val in sorted_impacts[:7]:
        shap_val     = float(shap_val)
        impact_rs    = _price_impact(shap_val, base_price)
        direction    = "positive" if shap_val > 0 else "negative"
        impact_label = f"+₹{impact_rs:,.0f}" if shap_val > 0 else f"-₹{impact_rs:,.0f}"

        if impact_rs < 1000:
            continue

        icon = label = simple = detail = ""
        
        # ── Car Age ─────────────────────────────────
        if feature == 'Car Age':
            age = int(vals['Car Age'])
            label = f"Car Age ({age} yrs)"

            # Use absolute age thresholds first, SHAP only for direction/magnitude
            if age >= 10:
                icon   = "🔴"
                simple = (
                    f"At {age} years old, this car has seen heavy depreciation. "
                    f"Cars typically lose 60–70% of original value after 10 years."
                )
                detail = "Older vehicles carry higher maintenance risk and lower market demand — compressing resale value significantly."
            elif age >= 6:
                icon   = "🟡"
                simple = (
                    f"At {age} years, this car has depreciated moderately. "
                    f"Still a common and accepted age bracket in the Indian used car market."
                )
                detail = "Mid-life vehicles balance affordability and remaining usable lifespan."
            elif age >= 4:
                icon   = "🟡"
                simple = f"{age} years old — minor depreciation applies."
                detail = "Some depreciation at this age but well within acceptable range."
            else:  # age < 4
                icon   = "🟢"
                simple = (
                    f"Only {age} years old — relatively new. "
                    f"Buyers pay a premium for newer cars with more remaining lifespan."
                )
                detail = "Low age means less wear, lower near-term maintenance costs, and stronger buyer confidence."
        
        # ── Engine ──────────────────────────────────
        elif feature == 'Engine':
            cc    = int(vals['Engine'])
            label = f"Engine Size ({cc}cc)"
            if shap_val > 0.4:
                icon   = "🟢"
                simple = (
                    f"Large {cc}cc engine signals a premium SUV or luxury car — "
                    f"high buyer demand and strong resale value."
                )
                detail = "Larger displacement engines are associated with premium segments (SUVs, MPVs, luxury sedans) that hold value significantly better."
            elif shap_val > 0.1:
                icon   = "🟢"
                simple = f"Mid-size {cc}cc engine — good balance of performance and economy. Adds moderate value."
                detail = "Mid-displacement engines appeal to a wide buyer base, supporting healthy resale demand."
            elif shap_val < -0.4:
                icon   = "🔴"
                simple = (
                    f"Small {cc}cc engine places this firmly in the budget hatchback segment. "
                    f"Lower resale value, though fuel economy is a positive."
                )
                detail = "Sub-1000cc engines are associated with entry-level cars which depreciate faster in absolute rupee terms."
            elif shap_val < -0.1:
                icon   = "🟡"
                simple = f"{cc}cc engine is slightly below average for this market — moderate negative impact."
                detail = "Engine size is marginally below segment average, reducing perceived value slightly."
            else:
                icon   = "🟡"
                simple = f"{cc}cc engine — neutral impact on price in this segment."
                detail = "Engine size aligns with segment average."

        # ── Km Driven ───────────────────────────────
        elif feature == 'Km':
            km    = int(vals['Km'])
            km_yr = vals.get('Km Per Year', 0)
            label = f"Mileage ({km:,} km)"
            avg   = 15000
            if km_yr > 25000:
                icon   = "🔴"
                simple = (
                    f"{km:,} km over {int(vals.get('Car Age', 1))} years = "
                    f"{km_yr:,.0f} km/year — that's {km_yr/avg:.1f}x the personal-use "
                    f"average of 15,000 km/year. Strongly suggests commercial or taxi use."
                )
                detail = (
                    "Extremely high annual mileage accelerates engine wear, "
                    "suspension fatigue, and brake wear — significantly reducing "
                    "resale value and raising reliability concerns."
                )
                warnings.append(
                    "⚠️ High km/year detected ({:.0f} km/yr). This car may have been "
                    "used commercially (taxi/cab). Request a full service history and "
                    "get an independent mechanical inspection before purchasing.".format(km_yr)
                )
            elif km_yr > 18000:
                icon   = "🟡"
                simple = (
                    f"{km:,} km — slightly above average usage "
                    f"({km_yr:,.0f} km/yr vs the 15,000 km/yr average). Minor price impact."
                )
                detail = "Above-average usage increases buyer perception of wear risk, applying a small discount."
            elif shap_val < -0.05:
                icon   = "🟡"
                simple = f"{km:,} km — moderate mileage. Within acceptable range but nudges price slightly downward."
                detail = "Mileage is within normal range but buyers will factor it into negotiations."
            else:
                icon   = "🟢"
                simple = f"{km:,} km — low to moderate usage. Well received by buyers and supports asking price."
                detail = "Low mileage signals less wear and a longer remaining vehicle life — a clear positive."

        # ── Owner ────────────────────────────────────
        elif feature == 'Owner':
            owner_map  = {1: "First", 2: "Second", 3: "Third"}
            owner_text = owner_map.get(int(vals['Owner']), "Multiple")
            label      = f"Ownership ({owner_text} owner)"
            if vals['Owner'] == 1:
                icon   = "🟢"
                simple = (
                    "Single owner — highest buyer confidence. "
                    "First-owner cars typically command a 15–25% premium over second-owner equivalents."
                )
                detail = "Single ownership minimises unknown usage and maintenance risks, making this significantly more attractive to buyers."
            elif vals['Owner'] == 2:
                icon   = "🟡"
                simple = (
                    "Two owners — moderate reduction in value. "
                    "Second-owner cars are common and widely accepted in India. "
                    "Buyers will negotiate but won't walk away."
                )
                detail = "Second ownership introduces some unknown history from the first owner, applying a moderate but manageable discount."
            else:
                icon   = "🔴"
                simple = (
                    f"{owner_text} owner — multiple ownership significantly reduces "
                    f"buyer confidence and resale value."
                )
                detail = "Multiple owners raise concerns about maintenance consistency and overall vehicle care."

        # ── Transmission ─────────────────────────────
        elif feature == 'Manual Transmission':
            label = "Transmission"
            if vals['Manual Transmission'] == 1:
                icon   = "🟡"
                simple = (
                    "Manual gearbox — priced lower than automatic equivalent. "
                    "In India's city traffic, automatics are increasingly preferred post-2018. "
                    "Upside: lower maintenance costs benefit the buyer."
                )
                detail = (
                    "Automatic transmission commands a 20–40% premium in urban markets. "
                    "Manual cars remain popular in Tier 2/3 cities and among cost-conscious buyers."
                )
            else:
                icon   = "🟢"
                simple = (
                    "Automatic transmission — strong demand in metro cities "
                    "adds meaningful value to this car."
                )
                detail = "Automatic cars appeal to urban stop-go commuters. Higher demand translates directly to stronger resale pricing."

        # ── Fuel ────────────────────────────────────
        elif feature == 'Diesel':
            label = "Fuel Type"
            if vals['Diesel'] == 1:
                icon   = "🟢"
                simple = (
                    "Diesel engine — preferred for highway and high-mileage use. "
                    "SUVs and MPVs in diesel hold resale value particularly well."
                )
                detail = "Diesel's fuel efficiency advantage at high mileage makes it the preferred choice for long-distance buyers, supporting stronger resale."
            else:
                icon   = "🟡"
                simple = "Non-diesel fuel type. Petrol/CNG are preferred for city use with lower running costs."
                detail = "Petrol and CNG cars have cost advantages in city use but carry a mild resale disadvantage in SUV and MPV segments dominated by diesel."

        elif feature == 'Petrol':
            if abs(shap_val) < 0.05:
                continue
            label = "Fuel Type"
            if vals['Petrol'] == 1:
                icon   = "🟡"
                simple = (
                    "Petrol engine — ideal for city commutes, lower maintenance. "
                    "Slightly lower resale than diesel in larger segments, "
                    "but the preferred choice for hatchbacks and compact sedans."
                )
                detail = "Petrol cars dominate hatchback segment resale. In SUV segments, they face a mild discount vs diesel."
            else:
                icon   = "🟡"
                simple = "CNG fuel — lowest running costs in city use. Strong appeal for high-mileage urban drivers."
                detail = "CNG cars offer very low per-km costs but limited highway use due to refuelling infrastructure."

        # ── Brand ────────────────────────────────────
        elif feature == 'Brand':
            label = "Brand"
            if shap_val > 0.2:
                icon   = "🟢"
                simple = "Premium brand — strong resale value. Luxury brands (Mercedes-Benz, BMW, Audi) hold value exceptionally well in metro markets."
                detail = "Brand premium reflects reliability perception, service network quality, and social cachet — all supporting resale demand."
            elif shap_val > 0.05:
                icon   = "🟢"
                simple = "Good brand reputation adds moderate value. Brands like Toyota and Honda are trusted for long-term reliability."
                detail = "Reliability reputation and strong service networks positively influence buyer willingness to pay."
            elif shap_val < -0.1:
                icon   = "🟡"
                simple = "Brand has a moderate negative impact in this segment. Note: Maruti Suzuki's unmatched service network is a strong plus for budget buyers despite the lower sticker price."
                detail = "Some brands face price perception challenges in certain segments despite solid fundamentals."
            else:
                icon   = "🟡"
                simple = "Brand has a neutral to minor impact on this car's price."
                detail = "Brand effect is minimal for this specific model-segment combination."

        # ── Model ────────────────────────────────────
        elif feature == 'Base Model':
            label = "Model"
            if shap_val > 0.08:
                icon   = "🟢"
                simple = (
                    "Strong resale model — popular cars like Swift, Creta, and Innova "
                    "are always in demand and hold value exceptionally well."
                )
                detail = "High-volume popular models have deep secondary markets with many active buyers, supporting healthy price floors."
            elif shap_val > 0:
                icon   = "🟢"
                simple = "This model has decent resale demand in the used car market."
                detail = "Solid market demand keeps prices reasonably stable for this model."
            elif shap_val < -0.08:
                icon   = "🟡"
                simple = "This model has relatively lower resale demand vs top sellers. Expect buyers to negotiate harder."
                detail = "Lower-volume or discontinued models have shallower secondary markets, compressing resale prices."
            else:
                icon   = "🟡"
                simple = "Model has a neutral impact on price — in line with segment average."
                detail = "Model-specific demand is close to the market average."

        # ── Location ─────────────────────────────────
        elif feature == 'Location':
            label = "City Market"
            if shap_val > 0.08:
                icon   = "🟢"
                simple = "High-demand city — metros like Mumbai and Bangalore see stronger used car prices driven by higher purchasing power and demand."
                detail = "Urban density and higher income levels in Tier 1 cities support stronger buyer demand and pricing power."
            elif shap_val < -0.08:
                icon   = "🟡"
                simple = "Lower demand in this city — consider listing on national platforms (CarDekho, Cars24) to reach buyers across India."
                detail = "Smaller city markets have fewer active buyers, giving buyers more negotiating leverage."
            else:
                icon   = "🟡"
                simple = "City has a neutral impact — average demand market, in line with national pricing."
                detail = "This city's used car market is close to the national average."

        # ── Seats ────────────────────────────────────
        elif feature == 'Seats':
            seats = int(vals['Seats'])
            label = f"Seating ({seats} seats)"
            if seats >= 7 and shap_val > 0:
                icon   = "🟢"
                simple = f"{seats}-seater — family and MPV buyers pay a meaningful premium for extra seating capacity in India."
                detail = "7+ seat vehicles are in strong demand among Indian families. MPVs and full-size SUVs command a clear premium."
            elif shap_val < 0:
                icon   = "🟡"
                simple = f"{seats}-seat configuration has a slight negative impact in this segment."
                detail = "Seating capacity is below the segment preference, applying a minor discount."
            else:
                icon   = "🟡"
                simple = f"Standard {seats}-seat layout — neutral market impact."
                detail = "Standard seating, no significant premium or discount applied."

        else:
            continue

        price_breakdown.append({
            "label":        label,
            "impact_rs":    impact_rs,
            "impact_label": impact_label,
            "direction":    direction,
            "icon":         icon,
            "simple":       simple,
            "detail":       detail,
            "shap_val":     shap_val,
        })

        tech_table.append({
            "Feature":    feature,
            "SHAP":       round(shap_val, 4),
            "₹ Impact":   impact_label,
            "Effect":     "▲ Adds value" if shap_val > 0 else "▼ Reduces value",
        })

    # ── Deal Score (1–10) ────────────────────────────
    pos = sum(b['impact_rs'] for b in price_breakdown if b['direction'] == 'positive')
    neg = sum(b['impact_rs'] for b in price_breakdown if b['direction'] == 'negative')
    total = pos + neg
    ratio = pos / total if total > 0 else 0.5
    deal_score = max(1, min(10, round(ratio * 10)))

    # ── Summary ──────────────────────────────────────
    car_age = int(vals.get('Car Age', 10))
    if deal_score >= 7:
        summary = "✅ Strong value — more factors are working in this car's favour than against it."
    elif deal_score >= 5:
        summary = "ℹ️ Balanced valuation — mixed signals. Inspect carefully and negotiate on the weaker factors."
    else:
        summary = "⚠️ Several factors reduce this car's value. There is likely room to negotiate the price downward."

    if car_age <= 3:
        warnings.append(
            "⚠️ This is a 2021+ car. Our training data has limited coverage of very recent models "
            "— the actual market price may be higher than predicted. "
            "Cross-check on CarDekho or Cars24 for live comparable listings."
        )

    return {
        "predicted_price": predicted_price,
        "base_price":      base_price,
        "summary":         summary,
        "deal_score":      deal_score,
        "price_breakdown": price_breakdown,
        "warnings":        warnings,
        "tech_table":      tech_table,
    }