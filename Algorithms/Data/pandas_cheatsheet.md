# Pandas Interview Cheat Sheet

### 1. Data Cleaning

#### 1.0 IO Tools

```python
pd.read_csv("xxx.csv"
        
        , parse_dates=["datetime"] # specify date columns
        )
pd.read_json()

pd.DataFrame({'col1':[1, 2, np.NaN]}) # define a dataframe
```

#### 1.1 Initial Data Quality Checks
```python
# Quick data profiling
df.describe()  # Statistical summary
df.info()      # Data types and non-null counts
df.duplicated().sum()  # Count duplicates
df.isnull().sum()     # Count missing values per column

df.rename(columns={'col1':'col1_new_name'}, inplace=True) # rename columns
```

#### 1.2 Handling Missing Values & Duplicates
```python


# Handle missing values
df.fillna(0, inplace=True)  # Fill with zero. Default to false
df.fillna(method='ffill')  # Forward fill
df.fillna(method='bfill')  # Backward fill

# Coalesce with combine_first: Update nulls with value in the same location in other.
s1.combine_first(s2)

df.dropna()  # Remove rows with any missing values

```

#### 1.3 Data Type Conversions

```python
# DateTime conversions
df['date'] = pd.to_datetime(df['date'] ,errors='coerce') # pd.timestamp, ensure invaid dates become NaN
df['month'] = df['date'].dt.month # i.e. year, weekday, hour

# time delta
df['date'].max() - df['date'].min() # timedelta
pd.Timedelta('2min')
pd.Timedelta(days=1)

# Type conversion
df['column'] = df['column'].astype('int64') # i.e. 'category'
df['column'] = pd.to_numeric(df['column'], errors='coerce')
```

#### 1.4 String Operations & Cleaning
```python
# String cleaning methods
df['column'].str.lower()
df['column'].str.strip()
df['column'].str.contains('pattern')
df['column'].str.extract(r'(\d+)')  # Extract numbers
```

#### 1.5 Filtering & Assigning values

- Assigning new columns
```python
# series.where() method
miss['adj'] = miss['x'].where(miss['x'] >= 0, other=pd.NA).astype('Int64')

# DataFrame.assign()
tips.assign(tip_rate=tips["tip"] / tips["total_bill"]) 
```

- Filtering
  - use ```df.loc[mask, cols]``` to combine row filtering and column selection in one step, typically faster than ```df[mask][cols]```

```python
# boolean mask
df['A'].gt(0) # .lt()
df['A'].eq(0)

# Row filtering based on boolean conditions --> return a df
df[(df['A'] > 0) & (df['B'].isin(['x', 'y']))]

# filter on both rows and columns
df.loc[row_boolean_mask, [cols]]

# update column values for filtered rows
seat.loc[seat['id']%2==0, 'student'] = original_students.shift(1)

# multiple value filters using .loc[] 
order_summary.loc[order_summary['order_count']>=3] \ 
                 .loc[order_summary['year'].diff(periods=1)==1] \ # two consecutive years
                 .loc[order_summary['order_count'].shift(1)>=3] # 

# filter based on indices
df.iloc[1:2] # return the 2nd row as a df. iloc[1] will return single row as series
df.iloc[1:2, 0:3] # slice on both rows and cols
```
- Date range filtering
  
```python
df['event_date'].between("2013-10-01", "2013-10-03")
```

#### 1.6 Dropping & Sorting

```python

# drop columns
df.drop(columns=column_list)
df.drop(column_names, axis=1, inplace=True) # alternative

# drop rows
df.drop(index=[])
df.drop(labels, axis=0) 

# Remove duplicates from all or certain columns
df.drop_duplicates(subset=['col1', 'col2']) # Return DataFrame with duplicate rows removed. default to use all of the columns

# Sorting
df.sort_values(['col1', 'col2'], ascending=[True, False], inplace=True)
```


- Similar to ```RANK() OVER(PARTITION BY col1 ORDER BY col2)``` : ```rank()```

  - ```method=min```: Assigns the lowest rank in the group to all tied values. The next rank resumes after the tied positions, like 1,2,2,4 ...
  - ```method=dense```: like min, but rank always increases by 1 between groups. i.e. 1,2,2,3...
  - ```method=average```: (default) average rank of the group. can be non-integar. i.e. 1.5, 1.5, 3 ...
  - ```method=first```: ranks assigned in order they appear in the array, ensuring unique ranks for each row
```python
# method: How to rank the group of records that have the same value (i.e. ties)

dist['rnk'] = dist['salary'].rank(method='dense', ascending=False)

# in-group rank column
employee['department_rnk'] = employee.groupby(['Department'])['salary'].rank(method='dense',ascending=False)
```


#### 1.7 Merging & Joining Data

- Equivalent to JOIN: ```merge(), join()```

```python
# # by default inner join
pd.merge(df1, df2, left_on='col1', right_on='col2')

df1.merge(df2, how='left',left_index=True, right_index=True) # join on index

df1.join(df2, on='key_column')
# append an indicator column source_ind about the source of row (left only, both, etc.)
pd.merge(df1, df2, on='key_column', how='left', indicator='source_ind') 

# validate argument checks the uniqueness of merge keys to protect against memory overflows and unexpected key duplication.
pd.merge(df1, df2, on="key", how="outer", validate="one_to_one") 

# cartesian product: create all combinations of two tables
result = pd.merge(left, right, how="cross")
```
- UNION ALL: ```pd.concat([df1, df2])```
- UNION: ```pd.concat([df1, df2]).drop_duplicates()```

- Left join a table based on the nearest key rather than equal keys:
    - df1 and df2 have to be sorted by both the ```by``` key and the ```on``` key before merge.
   ```python
    quotes = quotes.sort_values(['ticker','time'])
    trades = trades.sort_values(['ticker','time'])
    pd.merge_asof(trades, quotes, 
                on='time', # nearest match key
                by='ticker', # exact match key
                direction='backward', 
                tolerance=pd.Timedelta('2ms'))
   ```
- Interval mapping using cross-join
  
  merge() method is a vectorized operation. Hence it is significantly more performant than rowvise iteration using apply()

  ```python
  # cartesian product: create all combinations of two tables
  merged_df = values.merge(bands, how='cross')
  # Filter for the rows where 'value' falls within the band's range
  merged_df = merged_df[(merged_df['value'] >= merged_df['low']) & (merged_df['value'] <= merged_df['high'])]
  result = merged_df[['value', 'band']]
  ```


### 2. Data Transformation & Summarization

#### 2.1 Grouping & Aggregations

-  Rename the columns after aggregation &rarr; avoids the multi-level column index
```python
df.groupby(['A','B'] ,dropna=False).agg( # NA by default is dropped when groupby unless specified
    mean_x=('X','mean'),
    n=('X', 'size'),
    p90_y=('Y', lambda s: s.quantile(0.9)) 
).reset_index() # flatten the row index

```

| Method	| Applied To	| Logic Level	| Output Shape	| Best For... |
| -----| ---------| --------- | ---------- | --------
|map()	| Series	| Element-wise	| Same as input	| Dictionary lookups and simple value substitutions.
| agg()	| GroupBy / DF	| Group-wise	| Reduced (1 row/group)	| Summary statistics like sum, mean, or multiple stats at once.
| transform()	| GroupBy	| Group-wise	| Same as input	| Broadcasting group stats back to every row (e.g., centering data).
| apply()	| GroupBy / DF	| Flexible	| Flexible	| Complex custom logic that isn't covered by optimized built-in methods.

- agg → Reduce to scalar(s) per group

```python
# 1) agg: one row per group

# below basic aggregation results in multi-level column names
df.groupby('column_name').agg({
    'col1': 'sum',
    'col2': 'mean',
    'col3': ['min', 'max']
}).reset_index() # flatten the row index only

df.groupby('category').agg(lambda x: x.value_counts().index[0])  # Mode
```

- ```transform()``` for GroupBy or similar shape output → Return same shape as input (broadcasted features)
  - perferred method when output is in the same dimension
  - can be used to filter rows based on group stats

```python
# 2) transform: same length as original (great for assignment)

df['zscore'] = df.groupby('grp')['val'].transform(lambda s: (s - s.mean()) / s.std()) # normalize

# Using transform to apply a custom function with arguments to a column
def word_check(s, word):
    if ' '+word+' ' in s:
        return 1
    else:
        return 0
df['bull'] = df['content'].transform(word_check, word='bull')

# example: Keep only groups with at least 30 rows (by A,B).
keep = df.groupby(['A','B'])['X'].transform('size')>=30
df_filtered = df[keep]

```
- ```apply()``` for row-wise operations (multiple columns) → Flexible but slower; only when necessary

```python
# 3) apply: fully custom; output can be a scalar or the original dimension depending on the functions passed in.

## return a scalar per group
df.groupby('category').apply(lambda x: x['value'].max() - x['value'].min())  # Range

## return the original input dimension (transform method is prefered)  
def normalize(s):
    return (s - s.min()) / (s.max() - s.min())

applied = df.groupby('grp').apply(lambda g: g.assign(y_norm=normalize(g['y'])))

```
- ```map()``` for Element-wise Operations (Single Column). It DOES NOT work on a groupby object however. 
  
```python
# Define a function to categorize a value
def cost_category(value):
    if value < 500:
        return "Low"
    else:
        return "High"

# Create a new column using map() on the 'New_Column_Apply'
df['Category'] = df['New_Column_Apply'].map(cost_category)
```


- Top n rows per group using ```rank()```


#### 2.2 Window / Rolling Operations

- ```df[col].rolling(window, min_periods, closed)```
  - if window = a timedelta or string offset i.e. '30D' this is only valid with datetimelike indexes.
  - if window = an integer the index doesn't need to be date.
  - min_periods: min number of observations required to have a value; otherwise np.nan
  - closed: default is 'right' (fist, last], meaning the last point is included.

```python

df['3_day_avg'] = df['value'].rolling(window=3).mean()

# time-based rolling window within groups
roll_sum = (ts_df.sort_values(['id','ts']) # df has to be first sorted and 0-indexed
                .set_index('ts') # rolling method has to work on time based index
                .groupby('id')['value'] 
                .rolling(window='30D', min_periods=1).sum() # string offset '30D' has to be used with datetime index
                .reset_index() #  drop the extra indices (id, ts) used for rolling operation
                .rename(columns={'value':'roll_sum_30d'})
              )

ts_df = ts_df.merge(roll_sum, how='left', on = ['id','ts']) 

```

- ```df[col].diff(periods=1)```: equivalent to ```df[col] - df[col].shift(1)```.
  - Require sorting the dataframe first as diff() and shift() relies on row sequence.
  - periods lets you compare with the value n rows before.
  - The first periods rows become NaN.

```python
# Window functions: similar to LEAD()
df['previous_row'] = df['value'].shift(1)
df['next_row'] = df['value'].shift(-1)
df['value'].diff(periods=1) # diff to the prior row's value

# in-group window
order_summary['prev_year'] = order_summary.groupby('product_id')['year'].shift(1)

# gap in days for a user between two visits
activity['player_day_lag'] = activity.groupby('player_id')['event_date'].diff(periods=1).dt.days

df['cumulative_sum'] = df['value'].cumsum()
```

#### 2.3 Reshaping data

- "long" &rarr; "wide": ```df.pivot()``` vs ```df.pivot_table()```

    Both reshape a DataFrame by transforming rows into columns, but differ in their handling of duplicate entries and their aggregation capabilities. 

    - ```df.pivot(columns, index, values)```: requires the combination of index and columns to be unique. If there are duplicate entries for a given index/columns pair, pivot will raise a ValueError. It does not perform any aggregation. It simply rearranges the data. 

        &rarr; Use pivot when you are certain that the combination of index and columns will result in unique entries in the pivoted table and no aggregation is needed. 

    ```python
    df.pivot(index='date', columns='category', values='amount')
     ```
    - ```df.pivot_table(values, index, columns, aggfunc='mean')```: A more generalized version of pivot. It can handle duplicate entries for a given index/columns pair by applying an aggregation function (e.g., mean, sum, median). By default, it uses the mean. 

        &rarr; Use pivot_table when you anticipate duplicate entries for index/columns combinations and need to aggregate those values, or when you require more advanced summarization features like custom aggregation functions or margins. 

    ```python
    # aggregate over the values columns and include totals (margin=True). Fill missing with 0.
    df.pivot_table(index='date', columns='category', values=['sales','qty'], aggfunc='sum', fill_value=0, margins=True)
    # return MultiIndex dataframe
     ```

- "wide" &rarr; "long" with ```df.melt()``` vs ```df.stack()```

    The core difference is that melt() works on regular column names, whereas stack() is designed for hierarchical column indexing (MultiIndex).
    - ```df.melt(id_vars, value_vars, var_name, value_name)```: keep id_vars and unpivot value_vars. The melted columns are transformed into a new var_name column storing categories and a new value_name column storing values.

      ```python
      df.melt(id_vars=['date'], value_vars=['A', 'B'], var_name='category',value_name='value')
      ```
        &rarr; Best for datasets where column names do not follow a specific hierarchical pattern.

    - ```df.stack(level=-1)``` pivots a column level from a DataFrame's columns into its index, creating a MultiIndex. It is most powerful when used with a DataFrame that already has a MultiIndex on its columns, but it can also be used on a simple DataFrame. 

#### 2.4 Time series resampling

- ```df.resample(rule)```  is often used to change the frequency of a time series. It provides a time-based grouping, by using a string (e.g. 'M' for month, 'B' for business days, '5H',…) that defines the target frequency

    - Downsampling: decrease frequency
        
        it requires an aggregation function such as mean, max,… 
  ```python
  df.resample("ME").max() # aggregate values to monthly max with month-end as date index
  df.resample('M').mean()  # Monthly resampling
  ```    
    - Upsampling: crease frequency
      - ```.ffill()```: Forward fill, which uses the last known valid observation to fill gaps.
      - ```.bfill()```: Backward fill, which uses the next known valid observation to fill gaps.
  
  ```python
  # Resample to hourly and forward fill next 2 rows with the last known value 
  df_hourly = df_daily.resample('H').ffill(limit=2)
  ```  

