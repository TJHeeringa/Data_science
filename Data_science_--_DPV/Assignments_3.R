library(DBI)
library(RPostgreSQL)
library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)

SuperstoreSales_main <- read_delim(file = "SuperSales/SuperstoreSales_main.csv",
                                   delim = ";", col_names = TRUE, col_types = NULL)
speSuperstoreSales_manager <- read_delim(file = "SuperSales/SuperstoreSales_manager.csv",
                                         delim = ";", col_names = TRUE, col_types = NULL)
SuperstoreSales_returns <- read_delim(file = "SuperSales/SuperstoreSales_returns.csv",
                                      delim = ";", col_names = TRUE, col_types = NULL)
head(SuperstoreSales_main)
head(SuperstoreSales_manager)
head(SuperstoreSales_returns)

# make Product table ’product’:
product <- SuperstoreSales_main %>%
  select(`Product Name`, `Product Sub-Category`, `Product Category`) %>%
  rename(name = `Product Name`, sub_category=`Product Sub-Category`,category=`Product Category`) %>%
  arrange(name, sub_category, container) %>%
  group_by(name, sub_category, container) %>%
  distinct() %>%
  ungroup() %>%
  mutate(product_id = row_number())

# make Customer table ’customer’:
customer <- SuperstoreSales_main %>%
  select(`Customer Name`, `Customer Segment`, `Province`, `Region`) %>%
  rename(name = `Customer Name`, segment = `Customer Segment`, province = `Province`, region=`Region`) %>%
  arrange(name, segment) %>%
  group_by(name, segment) %>%
  distinct() %>%
  ungroup() %>%
  mutate(cust_id = row_number())

# make Sales table ’sales’:
sales <- SuperstoreSales_main %>%
  select(`Order Date`, `Sales`, `Order Quantity`, `Unit Price`, `Profit`, `Shipping Cost`, `Ship Date`, `Order ID`, `Product Name`, `Product Sub-Category`, `Product Category`, `Customer Name`, `Customer Segment`, `Province`, `Region`) %>%
  rename(order_date=`Order Date`, ship_date=`Ship Date`, profit=`Profit`) %>%
  arrange(order_date, ship_date, profit) %>%
  
  distinct() %>%
  ungroup() %>%
  mutate(sales_id = row_number())

sales <- sales %>%
  full_join(product, by = c("Product Name" = "name",
                            "Product Sub-Category" = "sub_category",
                            "Product Category" = "category")) %>%
  select( -`Product Name`, -`Product Sub-Category`, -`Product Category`)

sales <- sales %>%
  full_join(SuperstoreSales_returns, by = c("Order ID" = "Order ID")) %>%
  select( -`Order ID`)

sales$order_date = as.Date(as.character(sales$order_date), "%d/%m/%y",tz="UTC")
sales$ship_date = as.Date(as.character(sales$ship_date), "%d/%m/%y",tz="UTC")
sales$shipped_late<-ifelse(difftime(sales$ship_date,sales$order_date,units = "days") >2, 1, 0)

sales$Status[is.na(sales$Status)] <- 0
sales$returned<-ifelse(sales$Status == "Returned", 1, 0)

# Most valuable product
mvalprod <- sales %>% 
  select(sales, productid) %>%
  aggregate(by = list( sal$productid ), FUN = sum) %>%
  select(Group.1, sales) %>%
  rename(productid = Group.1) %>%
  arrange(-sales) %>%
  full_join(prod, by = c("productid" = "productid")) %>% 
  select(name, sales)



# plot of profit vs product
ggplot(data=head(arrange(sales, profit)), aes(x = reorder(`product_id`, `profit`), y = `profit`)) + geom_bar(stat = "identity")

ggplot(data=sales, aes(x = `product_id`, y = profit))+stat_summary(fun.y = sum)

# plot of shipped_late vs product
ggplot(data=sales, aes(x = `product_id`, y = shipped_late)) +
  stat_summary(fun.y = sum, na.rm = TRUE, group = 3, color = 'black', geom ='line')

# plot of returns vs product
ggplot(data=sales, aes(x = `product_id`, y = returned)) + geom_point() +
  stat_summary(fun.y = sum, na.rm = TRUE, group = 3, color = 'black', geom ='line')


drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, port = 5432, host = "castle.ewi.utwente.nl",
                 dbname = "dpv2a139", user = "dpv2a139", password = "Ah53VQLa",
                 options="-c search_path=ass3")
dbWriteTable(con, "product", value = product, overwrite = T, row.names = F)
dbWriteTable(con, "customer", value = customer, overwrite = T, row.names = F)
dbWriteTable(con, "sales", value = sales, overwrite = T, row.names = F)


dbListTables(con)
str(dbReadTable(con,"customer"))
str(dbReadTable(con,"sales"))
str(dbReadTable(con,"product"))

