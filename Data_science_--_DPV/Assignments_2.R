library(DBI)
library(RPostgreSQL)
library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)

self <- function(.data) {
  .data
}

data0 <- read_delim(file = "data/BI_Raw_Data_UTF.csv",
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
customer <- data0 %>%
  select(Customer_Name, Customer_Country) %>%
  rename(name = Customer_Name, country = Customer_Country) %>%
  arrange(name, country) %>%
  group_by(name, country) %>%
  distinct() %>%
  ungroup() %>%
  mutate(cust_id = row_number())

sales <- data0 %>%
  select(Order_ID, Order_Date_Day, Customer_Name, Customer_Country, Product_Name, Product_Category, Product_Order_Price_Total) %>%
  rename(order_id = Order_ID, order_date = Order_Date_Day, sales = Product_Order_Price_Total) %>%
  arrange(order_id) %>%
  full_join(product, by = c("Product_Name" = "name", "Product_Category" = "category")) %>%
  select( -Product_Name, -Product_Category) %>%
  full_join(customer, by = c("Customer_Name" = "name", "Customer_Country" = "country")) %>%
  select( -Customer_Name, -Customer_Country)


drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, port = 5432, host = "castle.ewi.utwente.nl",
                 dbname = "dpv2a139", user = "dpv2a139", password = "Ah53VQLa",
                 options="-c search_path=ass2")
dbWriteTable(con, "product", value = product, overwrite = T, row.names = F)
dbWriteTable(con, "customer", value = customer, overwrite = T, row.names = F)
dbWriteTable(con, "sales", value = sales, overwrite = T, row.names = F)


dbListTables(con)

cust <- dbReadTable(con,"customer")
sal <- dbReadTable(con,"sales")
prod <- dbReadTable(con,"product")


# Most valuable customer
mvalcust <- sal %>% 
  select(sales, cust_id) %>%
  aggregate(by = list( sal$cust_id ), FUN = sum) %>%
  select(Group.1, sales) %>%
  rename(cust_id = Group.1) %>%
  arrange(-sales) %>%
  full_join(cust, by = c("cust_id" = "cust_id")) %>% 
  select(name, sales)

first5valcust <- head(mvalcust,5)

# Most valuable product
mvalprod <- sal %>% 
  select(sales, productid) %>%
  aggregate(by = list( sal$productid ), FUN = sum) %>%
  select(Group.1, sales) %>%
  rename(productid = Group.1) %>%
  arrange(-sales) %>%
  full_join(prod, by = c("productid" = "productid")) %>% 
  select(name, sales)

first5valprod <- head(mvalprod, 5)

ggplot(data=first5valcust, aes(x = reorder(name, -sales), y = sales)) + geom_bar(stat= "identity") + ggtitle("Top 5 customers") + xlab("Customer") #+ theme(text = element_text(size=14))
ggplot(data=first5valprod, aes(x = reorder(name, -sales), y = sales)) + geom_bar(stat= "identity") + ggtitle("Top 5 products") + xlab("Product")

# ggplot(data=first5valcust, aes(x = name, y = sales))+geom_point() + geom_smooth(method= "lm", se = FALSE)
# ggplot(data=mvalprod, aes(x = name, y = sales))+geom_point() + geom_smooth(method= "lm", se = FALSE)

