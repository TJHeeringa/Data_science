library(DBI)
library(RPostgreSQL)
library(readr)
library(dplyr)
library(lubridate)

data0 <- read_delim(file = "data/BI_Raw_data.csv",
                    delim = ";", col_names = TRUE, col_types = NULL)
head(data0)

# make Product table ’product’:
product <- data0 %>%
  select(Product_Name, Product_Category) %>%
  rename(name = Product_Name, category = Product_Category) %>%
  arrange(name, category) %>%
  group_by(name, category) %>%
  distinct() %>%
  ungroup() %>%
  mutate(productid = row_number())

# make Customer table ’customer’:
sales <- data0 %>%
  # dan nog iets

sales <- sales %>%
  full_join(product, by = c("Product_Name" = "name",
                            "Product_Category" = "category")) %>%
  select( -Product_Name, -Product_Category)

drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, port = 5432, host = "castle.ewi.utwente.nl",
                 dbname = "dpv2a139 ", user = "dpv2a139 ", password = "Ah53VQLa",
                 options="-c search_path=ass2")
dbWriteTable(con, "product", value = product, overwrite = T, row.names = F)
dbWriteTable(con, "customer", value = customer, overwrite = T, row.names = F)
dbWriteTable(con, "sales", value = sales, overwrite = T, row.names = F)


dbListTables(con)
str(dbReadTable(con,"customer"))
str(dbReadTable(con,"sales"))
str(dbReadTable(con,"product"))
# or if a table is in schema ass2:
#dbGetQuery(con, "SELECT table_name FROM information_schema.tables
#                 WHERE table_schema=’ass2’") ## to get the tables from schema ass2
#str(dbReadTable(con, c("ass2", "sales")))

ggplot(data=, aes(x = , y = , col = ) + geom_point() + geom_smooth(method= "lm", se = False))