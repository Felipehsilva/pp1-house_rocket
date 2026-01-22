# House Rocket - Real Estate Analytics

This project aims to provide strategic insights for House Rocket, a digital platform whose business model involves purchasing and reselling properties using data analysis to maximize profit.

## Data Content
The dataset contains house sale prices for King County, including Seattle, for homes sold between May 2014 and May 2015.

| Column | Description |
| :--- | :--- |
| **id** | Unique ID for each home sold |
| **date** | Date of the home sale |
| **price** | Price of each home sold |
| **bedrooms** | Number of bedrooms |
| **bathrooms** | Number of bathrooms (0.5 indicates a toilet without a shower) |
| **sqft_living** | Square footage of the interior living space |
| **sqft_lot** | Square footage of the land space |
| **floors** | Number of floors |
| **waterfront** | Dummy variable for whether the property has a waterfront view |
| **view** | An index from 0 to 4 of how good the view was |
| **condition** | An index from 1 to 5 on the condition of the house |
| **grade** | An index from 1 to 13 for construction and design quality |
| **sqft_above** | Square footage of the interior housing space above ground level |
| **sqft_basement** | Square footage of the interior housing space below ground level |
| **yr_built** | The year the house was initially built |
| **yr_renovated** | The year of the house’s last renovation |
| **zipcode** | Zipcode area |
| **lat** | Latitude |
| **long** | Longitude |

---

### 1. Business Problem
House Rocket is a platform for buying and selling real estate. The company's CEO would like to maximize the company's profit by finding good deals (buying at the best price, finding the best properties, renovating if necessary, and selling more expensively). Currently, the process is all about using external sources to find good deals (Asking people, internet search, etc.), which is slow and limits the business scale.

### 2. Business Assumptions
* Properties with a "condition" score below 3 are considered for lower offers or renovation.
* Location (zipcode) is the primary factor for determining the median price of a region.
* Seasonality significantly impacts price and demand.
* Outliers in bedrooms and bathrooms are treated as data inconsistencies.

### 3. Solution Strategy
1.  **Data Collection:** Loading the King County dataset.
2.  **Data Cleaning:** Handling missing values and removing outliers (e.g., the house with 33 bedrooms).
3.  **Feature Engineering:** Creating new features like `price_m2`, `seasonality`, and `is_renovated`.
4.  **Hypothesis Testing:** Using exploratory data analysis to validate business assumptions.
5.  **Financial Recommendation:** Creating a "Buy/Not Buy" indicator and suggested sale prices.
6.  **Interactive Dashboard:** Developing a Streamlit app to allow the CEO to filter data by attributes.

### 4. Top 3 Data Insights
* **Insight 01:** Properties with a waterfront view are, on average, over 200% more expensive than those without.
* **Insight 02:** Houses with a construction grade above 10 have a much higher appreciation rate regardless of the year built.
* **Insight 03:** The best time to sell properties in this region is during the Summer, where prices reach their peak.

### 5. Business Results
The analysis identified specific properties with high profit potential. By following the "Buy" recommendations (properties in good condition below the regional median), the company can achieve:
* **Average Profit per Property:** 10% to 30% depending on the purchase timing.
* **Risk Mitigation:** Avoiding high-priced properties in regions with low demand.
  
Based on the analysis, the project identified:

* **Total properties to purchase:** [Insert Number]
* **Total investment required:** $[Insert Value]
* **Expected revenue from sales:** $[Insert Value]
* **Estimated Total Profit:** $[Insert Value]

### 6. Conclusions
Data-driven decision-making allows House Rocket to scale its operations. Instead of manual searches, the company can now use automated filters to find the best 10% of deals in the market instantly, ensuring a higher return on investment.

### 7. Lessons Learned
* Identifying data errors early (outliers) is crucial for accurate median calculations.
* The importance of translating technical metrics (like sqft) into business insights (profit margin).
* Streamlit is a powerful tool for delivering data to end-users in an accessible way.
