# Blind spot-check

Rate each output 1-5 for: plain language, accuracy against the numbers, usefulness of charts. Which system wrote which is revealed at the bottom.

## S03 - question: How risky is the counterparty?

### Output A
**How the CVA reflects counterparty risk for this swap**

CVA is the estimated cost today of the risk that the counterparty defaults before the swap ends. For this deal the CVA is $132, which is 0.0066% of the $2,000,000 notional, and the swap runs for 2 years.

The market price of insuring against default, the CDS spread, is 1.50%, implying a cumulative default probability of about 4.88% over the life. The highest average amount the counterparty could owe (peak EPE) is $6,825, occurring around 0.5 years, showing that most of the risk is concentrated early in the contract.

Caveats: The default risk is assumed constant over time, based on a single CDS spread. | Interest rates are modeled simply and can become negative, and only CVA is measured; other adjustments are not included.
Charts: default_probability (Look at the default probability curve, which shows a cumulative chance of 4.88% of default before the swap ends.), exposure_profile (Notice the exposure profile peaks at $6,825 average exposure at 0.5 years and a worst‑case exposure of $26,804 at 1 year.)

Plain language __/5   Accuracy __/5   Charts __/5

### Output B
**Counterparty risk snapshot for the swap**

CVA is the estimated cost today of the risk that the counterparty defaults before the swap ends. The CVA is $132, which is a tiny 0.0066% of the $2,000,000 notional, and the market prices the counterparty’s default risk at a 1.50% CDS spread. The model predicts a 4.88% chance of default over the 2 years and the highest average amount the counterparty could owe (peak EPE) is $6,825, occurring at 0.5 years.

These numbers show that while the potential loss at the worst average point is modest, there is a non‑negligible chance of default, and the early period contributes most to the CVA.

Caveats: The default risk is assumed constant over time from a single CDS spread. | Interest rates are modeled simply and can become negative, and only CVA is measured.
Charts: exposure_profile (The average exposure peaks at $6,825 around 0.5 years.), default_probability (The model estimates a 4.88% chance the counterparty defaults before the swap ends.)

Plain language __/5   Accuracy __/5   Charts __/5

## S01 - question: (none)

### Output A
**What the CVA result tells you**

CVA is the estimated cost today of the risk that the counterparty defaults before the swap ends. For this 5‑year swap the CVA comes out to $542, which is 0.0542% of the $1,000,000 notional, and reflects an 11.75% chance the counterparty will default at some point before maturity.

Most of the cost is driven by exposure around the middle of the life: the period ending at 2 years contributes the largest single share, about 15% of the total CVA. At that point the average exposure peaks at $11,030 and the worst‑case (95% quantile) exposure reaches $37,569.

Caveats: Counterparty default risk is assumed constant over time, based on a single CDS spread. | Interest rates are simulated with a simple one‑factor model, which can produce negative rates.
Charts: exposure_profile (The chart shows how the expected and potential exposure evolve over the 5‑year life of the swap.), cva_contribution (The chart highlights that the period ending at 2 years adds the biggest share, about 15%, to the total CVA.)

Plain language __/5   Accuracy __/5   Charts __/5

### Output B
**Estimated cost of counterparty default risk: $542**

CVA is the estimated cost today of the risk that the other side of the swap fails to pay before the swap ends. For this swap on $1,000,000, which runs for 5 years, it is $542, which is 0.0542% of the notional.

The amount the counterparty could owe us peaks at $11,030 on average, around 2 years in.

Caveats: Counterparty default risk is assumed constant over time, based on a single CDS spread. | Interest rates are simulated with a simple one-factor model, which can produce negative rates. | Only CVA is measured; other valuation adjustments (own default, funding, capital) are not.
Charts: exposure_profile (The blue line is the average amount the counterparty could owe us; the dashed line is a bad case.), cva_contribution (Each bar shows how much a period adds to the final CVA figure.)

Plain language __/5   Accuracy __/5   Charts __/5

## S02 - question: Why is the CVA this size?

### Output A
**Why the CVA is $2.25**

CVA is the estimated cost today of the risk that the counter‑party defaults before the swap ends. For this one‑year swap the CVA comes out to $2.25, which is only 0.0005% of the $500,000 notional.

The biggest driver is the period around 0.25 years, which alone adds 39% of the total CVA; at that point the modelled chance of default before maturity is 0.83%. The default probability of 0.83% reflects the cumulative chance that the counter‑party will fail at any time before the swap finishes. Since the exposure is highest near the start, the early contribution is larger than later periods.

Caveats: The default risk is assumed constant, based on a single CDS spread. | Interest rates are modelled with a simple one‑factor approach that can produce negative rates.
Charts: cva_contribution (Contribution of each time bucket to the total CVA.), exposure_profile (Average and worst‑case exposure over the life of the swap.)

Plain language __/5   Accuracy __/5   Charts __/5

### Output B
**Estimated cost of counterparty default risk: $2.25**

CVA is the estimated cost today of the risk that the other side of the swap fails to pay before the swap ends. For this swap on $500,000, which runs for 1 year, it is $2.25, which is 0.0005% of the notional.

The amount the counterparty could owe us peaks at $701 on average, around 0.25 years in.

Caveats: Counterparty default risk is assumed constant over time, based on a single CDS spread. | Interest rates are simulated with a simple one-factor model, which can produce negative rates. | Only CVA is measured; other valuation adjustments (own default, funding, capital) are not.
Charts: exposure_profile (The blue line is the average amount the counterparty could owe us; the dashed line is a bad case.), cva_contribution (Each bar shows how much a period adds to the final CVA figure.)

Plain language __/5   Accuracy __/5   Charts __/5

---
Key (do not read before rating): S03 A = pipeline; S03 B = baseline; S01 A = baseline; S01 B = pipeline; S02 A = baseline; S02 B = pipeline