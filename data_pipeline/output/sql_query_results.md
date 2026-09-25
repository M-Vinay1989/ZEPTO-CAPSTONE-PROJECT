# Module 1: SQL Query Results & Execution Output

## High Rated Books (Rating >= 4)

```sql
SELECT book_id, title, price_gbp, price_inr, rating
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC, title ASC;
```

### Query Output:

|   book_id | title                                                                    |   price_gbp |   price_inr |   rating |
|----------:|:-------------------------------------------------------------------------|------------:|------------:|---------:|
|        11 | 1,000 Places to See Before You Die                                       |       26.08 |     2751.44 |        5 |
|        46 | A Flight of Arrows (The Pathfinders #2)                                  |       55.53 |     5858.42 |        5 |
|        69 | A Spy's Devotion (The Regency Spies of London #1)                        |       16.97 |     1790.33 |        5 |
|        20 | A Time of Torment (Charlie Parker #14)                                   |       48.35 |     5100.92 |        5 |
|        64 | Between Shades of Gray                                                   |       20.79 |     2193.34 |        5 |
|        48 | Mrs. Houdini                                                             |       30.25 |     3191.38 |        5 |
|        30 | The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) |       52.3  |     5517.65 |        5 |
|        40 | The Girl You Lost                                                        |       12.29 |     1296.59 |        5 |
|        57 | The Passion of Dolssa                                                    |       28.32 |     2987.76 |        5 |
|        60 | The Red Tent                                                             |       35.66 |     3762.13 |        5 |
|        34 | The Silkworm (Cormoran Strike #2)                                        |       23.05 |     2431.78 |        5 |
|        59 | Voyager (Outlander #3)                                                   |       21.07 |     2222.89 |        5 |
|        29 | What Happened on Beale Street (Secrets of the South Mysteries #2)        |       25.37 |     2676.54 |        5 |
|        65 | While You Were Mine                                                      |       41.32 |     4359.26 |        5 |
|        52 | A Paris Apartment                                                        |       39.01 |     4115.55 |        4 |
|         8 | A Year in Provence (Provence #1)                                         |       56.88 |     6000.84 |        4 |
|        31 | Delivering the Truth (Quaker Midwife Mystery #1)                         |       20.89 |     2203.9  |        4 |
|         2 | Full Moon over Noahâs Ark: An Odyssey to Mount Ararat and Beyond                                                                          |       49.43 |     5214.86 |        4 |
|        68 | Lost Among the Living                                                    |       27.7  |     2922.35 |        4 |
|        23 | Murder at the 42nd Street Library (Raymond Ambler #1)                    |       54.36 |     5734.98 |        4 |
|        12 | Sharp Objects                                                            |       47.82 |     5045.01 |        4 |
|        49 | The Marriage of Opposites                                                |       28.08 |     2962.44 |        4 |
|        16 | The Murder of Roger Ackroyd (Hercule Poirot #4)                          |       44.1  |     4652.55 |        4 |
|        32 | The Mysterious Affair at Styles (Hercule Poirot #1)                      |       24.8  |     2616.4  |        4 |
|        39 | The No. 1 Ladies' Detective Agency (No. 1 Ladies' Detective Agency #1)   |       57.7  |     6087.35 |        4 |
|        14 | The Past Never Ends                                                      |       56.5  |     5960.75 |        4 |
|        56 | World Without End (The Pillars of the Earth #2)                          |       32.97 |     3478.34 |        4 |

---

## Top 5 Most Expensive Books in INR

```sql
SELECT book_id, title, price_gbp, price_inr
        FROM books
        ORDER BY price_inr DESC
        LIMIT 5;
```

### Query Output:

|   book_id | title                                                                  |   price_gbp |   price_inr |
|----------:|:-----------------------------------------------------------------------|------------:|------------:|
|        26 | Boar Island (Anna Pigeon #19)                                          |       59.48 |     6275.14 |
|        39 | The No. 1 Ladies' Detective Agency (No. 1 Ladies' Detective Agency #1) |       57.7  |     6087.35 |
|         8 | A Year in Provence (Provence #1)                                       |       56.88 |     6000.84 |
|        14 | The Past Never Ends                                                    |       56.5  |     5960.75 |
|        61 | The Last Painting of Sara de Vos                                       |       55.55 |     5860.52 |

---

## Distinct Categories in Database

```sql
SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name ASC;
```

### Query Output:

| category_name      |
|:-------------------|
| Historical Fiction |
| Mystery            |
| Travel             |

---

## Books in Price Range £20 to £40

```sql
SELECT book_id, title, price_gbp, price_inr, rating
        FROM books
        WHERE price_gbp BETWEEN 20.0 AND 40.0
        ORDER BY price_gbp ASC;
```

### Query Output:

|   book_id | title                                                                                             |   price_gbp |   price_inr |   rating |
|----------:|:--------------------------------------------------------------------------------------------------|------------:|------------:|---------:|
|        42 | Blood Defense (Samantha Brinkman #1)                                                              |       20.3  |     2141.65 |        3 |
|        51 | Love, Lies and Spies                                                                              |       20.55 |     2168.02 |        2 |
|        64 | Between Shades of Gray                                                                            |       20.79 |     2193.34 |        5 |
|        31 | Delivering the Truth (Quaker Midwife Mystery #1)                                                  |       20.89 |     2203.9  |        4 |
|        59 | Voyager (Outlander #3)                                                                            |       21.07 |     2222.89 |        5 |
|        34 | The Silkworm (Cormoran Strike #2)                                                                 |       23.05 |     2431.78 |        5 |
|         9 | The Road to Little Dribbling: Adventures of an American in Britain (Notes From a Small Island #2) |       23.21 |     2448.66 |        1 |
|        38 | Career of Evil (Cormoran Strike #3)                                                               |       24.72 |     2607.96 |        2 |
|        32 | The Mysterious Affair at Styles (Hercule Poirot #1)                                               |       24.8  |     2616.4  |        4 |
|        29 | What Happened on Beale Street (Secrets of the South Mysteries #2)                                 |       25.37 |     2676.54 |        5 |
|        37 | Extreme Prey (Lucas Davenport #26)                                                                |       25.4  |     2679.7  |        3 |
|        67 | Starlark                                                                                          |       25.83 |     2725.06 |        3 |
|        11 | 1,000 Places to See Before You Die                                                                |       26.08 |     2751.44 |        5 |
|        58 | Girl With a Pearl Earring                                                                         |       26.77 |     2824.24 |        1 |
|        22 | Poisonous (Max Revere Novels #3)                                                                  |       26.8  |     2827.4  |        3 |
|        27 | The Widow                                                                                         |       27.26 |     2875.93 |        2 |
|        68 | Lost Among the Living                                                                             |       27.7  |     2922.35 |        4 |
|        49 | The Marriage of Opposites                                                                         |       28.08 |     2962.44 |        4 |
|        57 | The Passion of Dolssa                                                                             |       28.32 |     2987.76 |        5 |
|        45 | Forever and Forever: The Courtship of Henry Longfellow and Fanny Appleton                         |       29.69 |     3132.3  |        3 |
|        48 | Mrs. Houdini                                                                                      |       30.25 |     3191.38 |        5 |
|         7 | The Great Railway Bazaar                                                                          |       30.54 |     3221.97 |        1 |
|        56 | World Without End (The Pillars of the Earth #2)                                                   |       32.97 |     3478.34 |        4 |
|        66 | The Secret Healer                                                                                 |       34.56 |     3646.08 |        3 |
|        24 | Most Wanted                                                                                       |       35.28 |     3722.04 |        3 |
|        60 | The Red Tent                                                                                      |       35.66 |     3762.13 |        5 |
|         4 | Vagabonding: An Uncommon Guide to the Art of Long-Term World Travel                               |       36.94 |     3897.17 |        2 |
|        47 | The House by the Lake                                                                             |       36.95 |     3898.23 |        1 |
|         5 | Under the Tuscan Sun                                                                              |       37.33 |     3938.31 |        3 |
|        55 | The Invention of Wings                                                                            |       37.34 |     3939.37 |        1 |
|        33 | In the Woods (Dublin Murder Squad #1)                                                             |       38.38 |     4049.09 |        2 |
|        10 | Neither Here nor There: Travels in Europe                                                         |       38.95 |     4109.23 |        3 |
|        52 | A Paris Apartment                                                                                 |       39.01 |     4115.55 |        4 |

---

## SQL JOIN: Books with Category Names

```sql
SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name ASC, b.title ASC;
```

### Query Output:

|   book_id | title                                                                                             | category_name      |   price_gbp |   price_inr |   rating |   in_stock |
|----------:|:--------------------------------------------------------------------------------------------------|:-------------------|------------:|------------:|---------:|-----------:|
|        46 | A Flight of Arrows (The Pathfinders #2)                                                           | Historical Fiction |       55.53 |     5858.42 |        5 |          1 |
|        52 | A Paris Apartment                                                                                 | Historical Fiction |       39.01 |     4115.55 |        4 |          1 |
|        69 | A Spy's Devotion (The Regency Spies of London #1)                                                 | Historical Fiction |       16.97 |     1790.33 |        5 |          1 |
|        64 | Between Shades of Gray                                                                            | Historical Fiction |       20.79 |     2193.34 |        5 |          1 |
|        45 | Forever and Forever: The Courtship of Henry Longfellow and Fanny Appleton                         | Historical Fiction |       29.69 |     3132.3  |        3 |          1 |
|        58 | Girl With a Pearl Earring                                                                         | Historical Fiction |       26.77 |     2824.24 |        1 |          1 |
|        63 | Girl in the Blue Coat                                                                             | Historical Fiction |       46.83 |     4940.56 |        2 |          1 |
|        50 | Glory over Everything: Beyond The Kitchen House                                                   | Historical Fiction |       45.84 |     4836.12 |        3 |          1 |
|        53 | Lilac Girls                                                                                       | Historical Fiction |       17.28 |     1823.04 |        2 |          1 |
|        68 | Lost Among the Living                                                                             | Historical Fiction |       27.7  |     2922.35 |        4 |          1 |
|        51 | Love, Lies and Spies                                                                              | Historical Fiction |       20.55 |     2168.02 |        2 |          1 |
|        48 | Mrs. Houdini                                                                                      | Historical Fiction |       30.25 |     3191.38 |        5 |          1 |
|        67 | Starlark                                                                                          | Historical Fiction |       25.83 |     2725.06 |        3 |          1 |
|        54 | The Constant Princess (The Tudor Court #1)                                                        | Historical Fiction |       16.62 |     1753.41 |        3 |          1 |
|        62 | The Guernsey Literary and Potato Peel Pie Society                                                 | Historical Fiction |       49.53 |     5225.42 |        1 |          1 |
|        47 | The House by the Lake                                                                             | Historical Fiction |       36.95 |     3898.23 |        1 |          1 |
|        55 | The Invention of Wings                                                                            | Historical Fiction |       37.34 |     3939.37 |        1 |          1 |
|        61 | The Last Painting of Sara de Vos                                                                  | Historical Fiction |       55.55 |     5860.52 |        2 |          1 |
|        49 | The Marriage of Opposites                                                                         | Historical Fiction |       28.08 |     2962.44 |        4 |          1 |
|        57 | The Passion of Dolssa                                                                             | Historical Fiction |       28.32 |     2987.76 |        5 |          1 |
|        60 | The Red Tent                                                                                      | Historical Fiction |       35.66 |     3762.13 |        5 |          1 |
|        66 | The Secret Healer                                                                                 | Historical Fiction |       34.56 |     3646.08 |        3 |          1 |
|        44 | Tipping the Velvet                                                                                | Historical Fiction |       53.74 |     5669.57 |        1 |          1 |
|        59 | Voyager (Outlander #3)                                                                            | Historical Fiction |       21.07 |     2222.89 |        5 |          1 |
|        65 | While You Were Mine                                                                               | Historical Fiction |       41.32 |     4359.26 |        5 |          1 |
|        56 | World Without End (The Pillars of the Earth #2)                                                   | Historical Fiction |       32.97 |     3478.34 |        4 |          1 |
|        43 | 1st to Die (Women's Murder Club #1)                                                               | Mystery            |       53.98 |     5694.89 |        1 |          1 |
|        15 | A Murder in Time                                                                                  | Mystery            |       16.64 |     1755.52 |        1 |          1 |
|        21 | A Study in Scarlet (Sherlock Holmes #1)                                                           | Mystery            |       16.73 |     1765.02 |        2 |          1 |
|        20 | A Time of Torment (Charlie Parker #14)                                                            | Mystery            |       48.35 |     5100.92 |        5 |          1 |
|        42 | Blood Defense (Samantha Brinkman #1)                                                              | Mystery            |       20.3  |     2141.65 |        3 |          1 |
|        26 | Boar Island (Anna Pigeon #19)                                                                     | Mystery            |       59.48 |     6275.14 |        3 |          1 |
|        38 | Career of Evil (Cormoran Strike #3)                                                               | Mystery            |       24.72 |     2607.96 |        2 |          1 |
|        31 | Delivering the Truth (Quaker Midwife Mystery #1)                                                  | Mystery            |       20.89 |     2203.9  |        4 |          1 |
|        37 | Extreme Prey (Lucas Davenport #26)                                                                | Mystery            |       25.4  |     2679.7  |        3 |          1 |
|        25 | Hide Away (Eve Duncan #20)                                                                        | Mystery            |       11.84 |     1249.12 |        1 |          1 |
|        13 | In a Dark, Dark Wood                                                                              | Mystery            |       19.63 |     2070.96 |        1 |          1 |
|        33 | In the Woods (Dublin Murder Squad #1)                                                             | Mystery            |       38.38 |     4049.09 |        2 |          1 |
|        24 | Most Wanted                                                                                       | Mystery            |       35.28 |     3722.04 |        3 |          1 |
|        23 | Murder at the 42nd Street Library (Raymond Ambler #1)                                             | Mystery            |       54.36 |     5734.98 |        4 |          1 |
|        28 | Playing with Fire                                                                                 | Mystery            |       13.71 |     1446.41 |        3 |          1 |
|        22 | Poisonous (Max Revere Novels #3)                                                                  | Mystery            |       26.8  |     2827.4  |        3 |          1 |
|        12 | Sharp Objects                                                                                     | Mystery            |       47.82 |     5045.01 |        4 |          1 |
|        19 | Tastes Like Fear (DI Marnie Rome #3)                                                              | Mystery            |       10.69 |     1127.79 |        1 |          1 |
|        18 | That Darkness (Gardiner and Renner #1)                                                            | Mystery            |       13.92 |     1468.56 |        1 |          1 |
|        30 | The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1)                          | Mystery            |       52.3  |     5517.65 |        5 |          1 |
|        36 | The Cuckoo's Calling (Cormoran Strike #1)                                                         | Mystery            |       19.21 |     2026.66 |        1 |          1 |
|        35 | The Exiled                                                                                        | Mystery            |       43.45 |     4583.98 |        3 |          1 |
|        41 | The Girl In The Ice (DCI Erika Foster #1)                                                         | Mystery            |       15.85 |     1672.18 |        3 |          1 |
|        40 | The Girl You Lost                                                                                 | Mystery            |       12.29 |     1296.59 |        5 |          1 |
|        17 | The Last Mile (Amos Decker #2)                                                                    | Mystery            |       54.21 |     5719.16 |        2 |          1 |
|        16 | The Murder of Roger Ackroyd (Hercule Poirot #4)                                                   | Mystery            |       44.1  |     4652.55 |        4 |          1 |
|        32 | The Mysterious Affair at Styles (Hercule Poirot #1)                                               | Mystery            |       24.8  |     2616.4  |        4 |          1 |
|        39 | The No. 1 Ladies' Detective Agency (No. 1 Ladies' Detective Agency #1)                            | Mystery            |       57.7  |     6087.35 |        4 |          1 |
|        14 | The Past Never Ends                                                                               | Mystery            |       56.5  |     5960.75 |        4 |          1 |
|        34 | The Silkworm (Cormoran Strike #2)                                                                 | Mystery            |       23.05 |     2431.78 |        5 |          1 |
|        27 | The Widow                                                                                         | Mystery            |       27.26 |     2875.93 |        2 |          1 |
|        29 | What Happened on Beale Street (Secrets of the South Mysteries #2)                                 | Mystery            |       25.37 |     2676.54 |        5 |          1 |
|        11 | 1,000 Places to See Before You Die                                                                | Travel             |       26.08 |     2751.44 |        5 |          1 |
|         6 | A Summer In Europe                                                                                | Travel             |       44.34 |     4677.87 |        2 |          1 |
|         8 | A Year in Provence (Provence #1)                                                                  | Travel             |       56.88 |     6000.84 |        4 |          1 |
|         2 | Full Moon over Noahâs Ark: An Odyssey to Mount Ararat and Beyond                                                                                                   | Travel             |       49.43 |     5214.86 |        4 |          1 |
|         1 | It's Only the Himalayas                                                                           | Travel             |       45.17 |     4765.44 |        2 |          1 |
|        10 | Neither Here nor There: Travels in Europe                                                         | Travel             |       38.95 |     4109.23 |        3 |          1 |
|         3 | See America: A Celebration of Our National Parks & Treasured Sites                                | Travel             |       48.87 |     5155.78 |        3 |          1 |
|         7 | The Great Railway Bazaar                                                                          | Travel             |       30.54 |     3221.97 |        1 |          1 |
|         9 | The Road to Little Dribbling: Adventures of an American in Britain (Notes From a Small Island #2) | Travel             |       23.21 |     2448.66 |        1 |          1 |
|         5 | Under the Tuscan Sun                                                                              | Travel             |       37.33 |     3938.31 |        3 |          1 |
|         4 | Vagabonding: An Uncommon Guide to the Art of Long-Term World Travel                               | Travel             |       36.94 |     3897.17 |        2 |          1 |

---

