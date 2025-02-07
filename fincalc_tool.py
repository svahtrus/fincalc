import streamlit as st
import pandas as pd
import altair as alt

## täiendavad ideed:
#1 10-aastane horisont
#2 lisada debt to EBITDA (EBITDA = NOPLAT + amort ja depr)
#3 lisada teatud näitajate standardhälve arvutamise tööriist

def calculate_operating_cash(raha, müügitulu, industry_operating_cash_pct):
    """
    Calculate Operating Cash as the minimum of:
    (a) raha
    (b) müügitulu * industry operating cash %
    """
    industry_operating_cash_decimal = industry_operating_cash_pct / 100.0
    return min(raha, müügitulu * industry_operating_cash_decimal)

def calculate_operating_current_assets(operating_cash, varud, nõuded_ostjate_vastu):
    """
    Calculate Operating Current Assets as the sum of:
    1) Operating Cash
    2) Varud
    3) Nõuded ostjate vastu
    """
    return operating_cash + varud + nõuded_ostjate_vastu

def calculate_invested_capital(oca, row):
    return (
        oca
        - row["võlad tarnijatele jm"]
        - row["lepingukohustused"]
        - row["tagasimaksekohustis"]
        - row["muud op kohustused"]
        + row["materiaalsed põhivarad"]
        + row["immateriaalsed põhivarad"]
        + row["kasutusõiguse varad"]
        + row["muud op varad"]
    )
def noplat(müügitulu, otsekulud, turustuskulud, üldhalduskulud, muud_ärikulud):
    return müügitulu - otsekulud - turustuskulud - üldhalduskulud - muud_ärikulud

def calculate_average_invested_capital(current_invested_capital, previous_invested_capital):
    """
    Calculate the average invested capital as the average of the current and previous year's invested capital.
    If previous_invested_capital is None (e.g., first year), simply return current_invested_capital.
    """
    if previous_invested_capital is None:
        return current_invested_capital
    else:
        return (current_invested_capital + previous_invested_capital) / 2

def main():
    st.title("Financial Metrics Calculator")
    st.write("Enter the financial data for up to **10 years** below:")

    # Define the columns for the DataFrame
    columns = [
        "Year",
        "raha",
        "nõuded ostjate vastu",
        "varud",
        "materiaalsed põhivarad",
        "immateriaalsed põhivarad",
        "kasutusõiguse varad",
        "muud op varad",
        "võlad tarnijatele jm",
        "lepingukohustused",
        "tagasimaksekohustis",
        "muud op kohustused",
        "lühiajalised laenud",
        "pikaajalised laenud",
        "omakapital",
        "müügitulu",
        "otsekulud",
        "turustuskulud",
        "üldhalduskulud",
        "muud ärikulud",
        "põhivarade amort ja depr",
        "industry operating cash %"
    ]

    # Initialize an empty DataFrame with 10 rows
    data = {column: ["" if column == "Year" else 0.0 for _ in range(10)] for column in columns}
    df = pd.DataFrame(data)

    # Define column_config for later use

    column_config={
            "Year": st.column_config.TextColumn(
                "Year",
                help="Enter the fiscal year (e.g., 2021)"
            ),
            "raha": st.column_config.NumberColumn(
                "raha",
                format="%.2f",
                help="Enter raha value"
            ),
            "nõuded ostjate vastu": st.column_config.NumberColumn(
                "nõuded ostjate vastu",
                format="%.2f",
                help="Enter nõuded ostjate vastu value"
            ),
            "varud": st.column_config.NumberColumn(
                "varud",
                format="%.2f",
                help="Enter varud value"
            ),
            "materiaalsed põhivarad": st.column_config.NumberColumn(
                "materiaalsed põhivarad",
                format="%.2f",
                help="Enter materiaalsed põhivarad value"
            ),
            "immateriaalsed põhivarad": st.column_config.NumberColumn(
                "immateriaalsed põhivarad",
                format="%.2f",
                help="Enter immateriaalsed põhivarad value"
            ),
            "kasutusõiguse varad": st.column_config.NumberColumn(
                "kasutusõiguse varad",
                format="%.2f",
                help="Enter kasutusõiguse varad value"
            ),
            "muud op varad": st.column_config.NumberColumn(
                "muud op varad",
                format="%.2f",
                help="Enter muud op varad value"
            ),
            "võlad tarnijatele jm": st.column_config.NumberColumn(
                "võlad tarnijatele jm",
                format="%.2f",
                help="Enter võlad tarnijatele ja muud võlad value"
            ),
            "lepingukohustused": st.column_config.NumberColumn(
                "lepingukohustused",
                format="%.2f",
                help="Enter lepingukohustused value"
            ),
            "tagasimaksekohustis": st.column_config.NumberColumn(
                "tagasimaksekohustis",
                format="%.2f",
                help="Enter tagasimaksekohustis value"
            ),
            "muud op kohustused": st.column_config.NumberColumn(
                "muud op kohustused",
                format="%.2f",
                help="Enter muud op kohustused value"
            ),
            "lühiajalised laenud": st.column_config.NumberColumn(
                "lühiajalised laenud",
                format="%.2f",
                help="Enter lühiajalised laenud value"
            ),
            "pikaajalised laenud": st.column_config.NumberColumn(
                "pikaajalised laenud",
                format="%.2f",
                help="Enter pikaajalised laenud value"
            ),
            "omakapital": st.column_config.NumberColumn(
                "omakapital",
                format="%.2f",
                help="Enter omakapital value"
            ),
            "müügitulu": st.column_config.NumberColumn(
                "müügitulu",
                format="%.2f",
                help="Enter müügitulu value"
            ),
            "otsekulud": st.column_config.NumberColumn(
                "otsekulud",
                format="%.2f",
                help="Enter müüdud kaupade ja teenuste kulud value"
            ),
            "turustuskulud": st.column_config.NumberColumn(
                "turustuskulud",
                format="%.2f",
                help="Enter turustuskulud value"
            ),
            "üldhalduskulud": st.column_config.NumberColumn(
                "üldhalduskulud",
                format="%.2f",
                help="Enter üldhalduskulud value"
            ),
            "muud ärikulud": st.column_config.NumberColumn(
                "muud ärikulud",
                format="%.2f",
                help="Enter muud ärikulud value"
            ),
            "põhivarade amort ja depr": st.column_config.NumberColumn(
                "põhivarade amort ja depr",
                format="%.2f",
                help="Enter põhivarade kulum ja väärtuse langus value"
            ),
          
               "industry operating cash %": st.column_config.NumberColumn(
                "industry operating cash %",
                format="%.2f",
                help="Enter industry operating cash percentage (e.g., 20.0 for 20%)"
            ),
    }
    
    edited_df=df.copy()

    # Before the st.experimental_data_editor block
    st.subheader("Load Previously Saved Data")
    
    uploaded_file = st.file_uploader("Upload Financial Data CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            uploaded_df = pd.read_csv(uploaded_file, dtype={"Year": str})
            uploaded_df["Year"] = uploaded_df["Year"].fillna("").astype(str)
            
            # Validate that the uploaded CSV has the correct columns
            expected_columns = [
                "Year",
                "raha",
                "nõuded ostjate vastu",
                "varud",
                "materiaalsed põhivarad",
                "immateriaalsed põhivarad",
                "kasutusõiguse varad",
                "muud op varad",
                "võlad tarnijatele jm",
                "lepingukohustused",
                "tagasimaksekohustis",
                "muud op kohustused",
                "lühiajalised laenud",
                "pikaajalised laenud",
                "omakapital",
                "müügitulu",
                "otsekulud",
                "turustuskulud",
                "üldhalduskulud",
                "muud ärikulud",
                "põhivarade amort ja depr",
                "industry operating cash %"
            ]          
            # For each expected column, check if it exists; if not, add it with a default value
            for col in expected_columns:
                if col not in uploaded_df.columns:
                # If it's the "Year" column, use an empty string, otherwise 0.0
                    uploaded_df[col] = "" if col == "Year" else 0.0

            # Optionally, reorder the DataFrame columns to match expected_columns
            uploaded_df = uploaded_df[expected_columns]

            st.success("Data loaded successfully!")
                
             # Update the edited_df with uploaded data
            edited_df = st.data_editor(
                uploaded_df,
                num_rows=10,
                column_config=column_config,  # Ensure this matches your existing column_config
                height=600,
                key="financial_data_uploaded"  # Use a different key to avoid conflicts
            )
            # Remove rows that are completely empty (e.g., where "Year" is blank)
            edited_df = edited_df[edited_df["Year"].astype(str).str.strip() != ""]

        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        # If no file is uploaded, display the default editor
        edited_df = st.data_editor(
        df,
        num_rows=10,
        column_config=column_config,
        height=600,
        key="financial_data"
        )   

         # Remove rows that are completely empty (e.g., where "Year" is blank)
        edited_df = edited_df[edited_df["Year"].astype(str).str.strip() != ""]
    st.markdown("---")


    # After the st.experimental_data_editor block
    st.subheader("Save Your Input Data")
    
    # Provide a download button for the input data
    csv_input = edited_df.to_csv(index=True)
    st.download_button(
        label="Download Input Data as CSV",
        data=csv_input,
        file_name='financial_data_input.csv',
        mime='text/csv',
    )

    # Button to confirm or proceed with calculations
    if st.button("Calculate Metrics"):
        # Validate and process the data
        valid = True
        for idx, row in edited_df.iterrows():
            # Ensure Year is not empty
            if not row["Year"]:
                st.error(f"Year is missing in row {idx + 1}. Please enter the fiscal year.")
                valid = False
        if valid:
            st.subheader("Calculated Metrics per Year")

            prev_invested_capital = None

            results = []
            for idx, row in edited_df.iterrows():
                # Extract values
                year = row["Year"]
                raha = row["raha"]
                nõuded_ostjate_vastu = row["nõuded ostjate vastu"]
                varud = row["varud"]
                müügitulu = row["müügitulu"]
                industry_operating_cash_pct = row["industry operating cash %"]
                otsekulud = row["otsekulud"]
                turustuskulud = row["turustuskulud"]
                üldhalduskulud = row["üldhalduskulud"]
                muud_ärikulud = row["muud ärikulud"]
                põhivarade_amort_ja_depr = row["põhivarade amort ja depr"]
                lühiajalised_laenud = row["lühiajalised laenud"]
                pikaajalised_laenud = row["pikaajalised laenud"]
                omakapital = row["omakapital"]

                # Validate numeric inputs
                if any(pd.isna([raha, nõuded_ostjate_vastu, varud, müügitulu, industry_operating_cash_pct])):
                    st.error(f"Numeric inputs are missing in row {idx + 1}. Please fill all fields.")
                    valid = False
                    break

                # Calculate Operating Cash
                operating_cash = calculate_operating_cash(raha, müügitulu, industry_operating_cash_pct)

                # Calculate Operating Current Assets
                oca = calculate_operating_current_assets(operating_cash, varud, nõuded_ostjate_vastu)

                # Calculate Invested Capital
                invested_capital = (
                    oca
                    - row["võlad tarnijatele jm"]
                    - row["lepingukohustused"]
                    - row["tagasimaksekohustis"]
                    - row["muud op kohustused"]
                    + row["materiaalsed põhivarad"]
                    + row["immateriaalsed põhivarad"]
                    + row["kasutusõiguse varad"]
                    + row["muud op varad"]
                )

                # Calculate NOPLAT
                noplat_value = noplat(müügitulu, otsekulud, turustuskulud, üldhalduskulud, muud_ärikulud)

                # Calculate average invested capital
                avg_invested_capital = calculate_average_invested_capital(invested_capital, prev_invested_capital)
    
                # Store current invested capital for the next iteration
                prev_invested_capital = invested_capital
                
                # Calculate ROIC
                roic = (noplat_value / avg_invested_capital) if avg_invested_capital != 0 else 0
               
                # Append to results
                results.append({
                    "Year": year,
                    "Operating Cash": operating_cash,
                    "Operating Current Assets": oca,
                    "Invested Capital": invested_capital,
                    "Average Invested Capital": avg_invested_capital,
                    "NOPLAT": noplat_value,  
                    "ROIC": roic,
                    "Revenues": müügitulu,
                    "põhivarade amort ja depr": põhivarade_amort_ja_depr,
                    "muud ärikulud": muud_ärikulud,
                    "üldhalduskulud": üldhalduskulud,
                    "turustuskulud": turustuskulud,
                    "otsekulud": otsekulud,
                    "lühiajalised laenud": lühiajalised_laenud,
                    "pikaajalised laenud": pikaajalised_laenud,
                    "omakapital": omakapital

                })
            
            if valid:
                # Convert results to DataFrame
                results_df = pd.DataFrame(results)

                #calculate Deltas
                results_df["Change in Invested Capital"] = results_df["Invested Capital"].diff() + results_df["põhivarade amort ja depr"]

                results_df["FCF"] = results_df["NOPLAT"] + results_df["põhivarade amort ja depr"] - results_df["Change in Invested Capital"]

                #calculate ratios
                results_df["Ratio NOPLAT"] = results_df["NOPLAT"] / results_df["Revenues"]
                results_df["Ratio muud ärikulud"] = results_df["muud ärikulud"] / results_df["Revenues"]
                results_df["Ratio üldhalduskulud"] = results_df["üldhalduskulud"] / results_df["Revenues"]
                results_df["Ratio turustuskulud"] = results_df["turustuskulud"] / results_df["Revenues"]
                results_df["Ratio otsekulud"] = results_df["otsekulud"] / results_df["Revenues"]

                #calculate operating margin
                results_df["Operating Margin"] = results_df["NOPLAT"] / results_df["Revenues"]

                #calculate Revenues to invested capital
                results_df["Revenues to Invested Capital"] = results_df["Revenues"] / results_df["Average Invested Capital"]

                # Calculate Revenue Growth as a percentage
                results_df["Revenue Growth"] = (results_df["Revenues"].diff() / results_df["Revenues"].shift(1)) * 100

                # Calculate Debt to Equity ratio
                results_df["D/E ratio"] = (results_df["lühiajalised laenud"] + results_df["pikaajalised laenud"]) / results_df["omakapital"]

                # Calculate Debt to EBITDA ratio
                results_df["D/EBITDA"] = (results_df["lühiajalised laenud"] + results_df["pikaajalised laenud"]) / (results_df["NOPLAT"] + results_df["põhivarade amort ja depr"])

                #Compute the returns for all numeric columns
                returns = results_df.select_dtypes(include=['number']).pct_change()
                returns["Year"] = "Returns"

                #Compute the std_devs for all numeric columns
                std_devs = std_devs = returns.std(numeric_only=True)
                std_devs["Year"] = "Volatility"

                # Compute the averages for all numeric columns
                averages = results_df.mean(numeric_only=True)
                averages["Year"] = "Average"  # Label the averages row

                # Append the averages and Std Dev row to the DataFrame
                results_df_with_avg = pd.concat([results_df, pd.DataFrame([averages]), pd.DataFrame([std_devs])], ignore_index=True)

                # Format the numbers with commas and two decimal places or %
                results_df_with_avg["Operating Cash"] = results_df_with_avg["Operating Cash"].map("{:,.2f}".format)
                results_df_with_avg["Operating Current Assets"] = results_df_with_avg["Operating Current Assets"].map("{:,.2f}".format)
                results_df_with_avg["Invested Capital"] = results_df_with_avg["Invested Capital"].map("{:,.2f}".format)
                results_df_with_avg["Average Invested Capital"] = results_df_with_avg["Average Invested Capital"].map("{:,.2f}".format)
                results_df_with_avg["NOPLAT"] = results_df_with_avg["NOPLAT"].map("{:,.2f}".format)
                results_df_with_avg["ROIC"] = results_df_with_avg.apply(
                    lambda row: f"{row['ROIC']:.2f}" if row.name == results_df_with_avg.index[-1] 
                    else f"{row['ROIC']*100:.2f}%", axis=1
                )
                results_df_with_avg["Revenues"] = results_df_with_avg["Revenues"].map("{:,.2f}".format)
                results_df_with_avg["Operating Margin"] = results_df_with_avg.apply(
                    lambda row: f"{row['Operating Margin']:.2f}" if row.name == results_df_with_avg.index[-1] 
                    else f"{row['Operating Margin']*100:.2f}%", axis=1
                )
                results_df_with_avg["Change in Invested Capital"] = results_df_with_avg["Change in Invested Capital"].map(lambda x: "{:,.2f}".format(x) if pd.notna(x) else "")
                results_df_with_avg["põhivarade amort ja depr"] = results_df_with_avg["põhivarade amort ja depr"].map("{:,.2f}".format)
                results_df_with_avg["FCF"] = results_df_with_avg["FCF"].map("{:,.2f}".format)
                results_df_with_avg["Revenue Growth"] = results_df_with_avg.apply(
                    lambda row: f"{row['Revenue Growth']:.2f}" if row.name == results_df_with_avg.index[-1] 
                    else f"{row['Revenue Growth']:.2f}%", axis=1
                )
                results_df_with_avg["D/E ratio"] = results_df_with_avg ["D/E ratio"].map("{:,.2f}".format)
                results_df_with_avg["D/EBITDA"] = results_df_with_avg ["D/EBITDA"].map("{:,.2f}".format)

                # List the columns you want to display
                display_columns = [
                    "Year",
                    "Revenues",
                    "NOPLAT",
                    "Average Invested Capital",
                    "ROIC",
                    "Operating Margin",
                    "Revenue Growth",
                    "põhivarade amort ja depr",
                    "Change in Invested Capital",
                    "FCF",
                    "D/E ratio",
                    "D/EBITDA"
                ]

                # Filter the DataFrame to include only those columns
                filtered_results = results_df_with_avg[display_columns]

                # Display the filtered results
                st.table(filtered_results)

                csv_df = filtered_results.copy()
                csv_df = csv_df.applymap(lambda x: str(x).replace(",", "") if isinstance(x, str) else x)
                csv = csv_df.to_csv(index=False)

                st.download_button(
                    label="Download Metrics as CSV",
                    data=csv,
                    file_name='financial_metrics.csv',
                    mime='text/csv'
                )
            
                # Exclude the average row from charting.
                results_df_no_avg = results_df[results_df["Year"] != "Average"].copy()

                results_df_no_avg["ROIC_numeric"] = results_df_no_avg["ROIC"].apply(lambda x: f"{x * 100}")
                results_df_no_avg["Revenue_Growth_numeric"] = pd.to_numeric(results_df_no_avg["Revenue Growth"].astype(str).str.replace("%", ""), errors="coerce")

                #Create a chart for ROIC and Revenue Growth
                chart_df = results_df_no_avg[["Year", "ROIC_numeric", "Revenue_Growth_numeric"]].copy()
                chart_df = chart_df.rename(columns={
                    "ROIC_numeric" : "ROIC",
                    "Revenue_Growth_numeric" : "Revenue Growth"
                })

                # Melt the DataFrame so that each row represents one metric for a given year.
                chart_data = chart_df.melt(id_vars="Year", var_name="Metric", value_name="Value")

                # Create the line chart.
                chart = alt.Chart(chart_data).mark_line(point=True).encode(
                    x=alt.X("Year:N", title="Year"),
                    y=alt.Y("Value:Q", title="Percentage (%)"),
                    color=alt.Color("Metric:N", title="Metric"),
                    tooltip=["Year", "Metric", "Value"]
                ).properties(
                    title="ROIC and Revenue Growth Over the Years",
                    width=600,
                    height=400
                )

                # Display the chart in Streamlit.
                st.altair_chart(chart, use_container_width=True)

            #CHART for operating margin/revenues to invested capital
                # Convert the formatted "Operating Margin" (e.g. "15.00%") to a numeric value (e.g. 0.15)
                results_df_no_avg["Operating_Margin_numeric"] = pd.to_numeric(results_df_no_avg["Operating Margin"].astype(str).str.replace("%", ""), errors="coerce")

                # Ensure "Revenues to Invested Capital" is numeric.
                # (Assuming you have a column named "Revenues to Invested Capital".)
                results_df_no_avg["Revenues_to_Invested_Capital"] = pd.to_numeric(results_df_no_avg["Revenues to Invested Capital"], errors="coerce")

                #Compute averages of both values
                avg_operating_margin = results_df_no_avg["Operating_Margin_numeric"].mean()
                avg_revenues_to_invested_capital = results_df_no_avg["Revenues_to_Invested_Capital"].mean()

                # Get the unique years (assumed to be stored as strings)
                years = sorted(results_df_no_avg["Year"].unique())

                avg_line_data_margin = pd.DataFrame({
                    "Year": years,
                    "Average_Operating_Margin": [avg_operating_margin] * len(years)
                })

                avg_line_data_revenues = pd.DataFrame({
                    "Year": years,
                    "Average_Revenues_to_Invested_Capital": [avg_revenues_to_invested_capital] * len(years)
                })

                # Chart for Operating Margin (as a percentage)
                line_chart_margin = alt.Chart(results_df_no_avg).mark_line(point=True, color='blue').encode(
                    x=alt.X("Year:N", title="Year"),
                    y=alt.Y("Operating_Margin_numeric:Q", 
                            axis=alt.Axis(title="Operating Margin", format="%"))
                )

                # Chart for Revenues to Invested Capital (numerical value)
                line_chart_revenues = alt.Chart(results_df_no_avg).mark_line(point=True, color='red').encode(
                    x=alt.X("Year:N", title="Year"),
                    y=alt.Y("Revenues_to_Invested_Capital:Q", title="Revenues to Invested Capital")
                )

                # Dashed line for average Operating Margin
                avg_line_margin = alt.Chart(avg_line_data_margin).mark_line(strokeDash=[4,2], color='blue').encode(
                    x=alt.X("Year:N", title="Year"),
                    y=alt.Y("Average_Operating_Margin:Q")
                )

                # Dashed line for average Revenues to Invested Capital
                avg_line_revenues = alt.Chart(avg_line_data_revenues).mark_line(strokeDash=[4,2], color='red').encode(
                    x=alt.X("Year:N", title="Year"),
                    y=alt.Y("Average_Revenues_to_Invested_Capital:Q")
                )

                # Layer for Operating Margin (regular line + average dashed line)
                layer_margin = alt.layer(line_chart_margin, avg_line_margin)

                # Layer for Revenues to Invested Capital (regular line + average dashed line)
                layer_revenues = alt.layer(line_chart_revenues, avg_line_revenues)

                dual_axis_chart = alt.layer(
                    layer_margin,
                    layer_revenues
                ).resolve_scale(
                    y='independent'
                ).properties(
                    title="Operating Margin (%) and Revenues to Invested Capital w Averages",
                    width=600,
                    height=400
                )

                st.altair_chart(dual_axis_chart, use_container_width=True)

                # Prepare chart data and create a stacked bar chart
                results_df_no_avg = pd.DataFrame(results)
                chart_df = results_df_no_avg[["Year", "NOPLAT", "muud ärikulud", "üldhalduskulud", "turustuskulud", "otsekulud"]].copy()
                chart_data = chart_df.melt(id_vars="Year", var_name="Metric", value_name="Value")
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("Year:N", title="Year"),
                    y=alt.Y("sum(Value):Q", title="Value"),
                    color=alt.Color("Metric:N", legend=alt.Legend(title="Metric")),
                    tooltip=["Year", "Metric", "Value"]
                ).properties(
                    title="Yearly Breakdown of revenues",
                    width=600,
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)    



                # Select the Year and the five ratio columns
                ratio_cols = ["Ratio NOPLAT", "Ratio muud ärikulud", "Ratio üldhalduskulud", "Ratio turustuskulud", "Ratio otsekulud"]
                chart_df = results_df[["Year"] + ratio_cols].copy()
                chart_data = chart_df.melt(id_vars="Year", var_name="Metric", value_name="Ratio")
                chart = alt.Chart(chart_data).mark_line(point=True).encode(
                    x=alt.X("Year:N", title="Year"),
                    y=alt.Y("Ratio:Q", title="Ratio (fraction)"),
                    color=alt.Color("Metric:N", title="Metric"),
                    tooltip=["Year", "Metric", "Ratio"]
                ).properties(
                    title="Year-on-Year Ratios vs Revenues",
                    width=600,
                    height=400
                )

                st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
