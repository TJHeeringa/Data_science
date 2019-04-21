library(DBI)
library(RPostgreSQL)
library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(stringr)

SuperstoreSales_main <- read.csv2(file = "SuperSales/SuperstoreSales_main.csv", header = TRUE, stringsAsFactors = FALSE)
SuperstoreSales_manager <- read.csv2(file = "SuperSales/SuperstoreSales_manager.csv", header = TRUE, stringsAsFactors = FALSE)
SuperstoreSales_returns <- read.csv2(file = "SuperSales/SuperstoreSales_returns.csv", header = TRUE, stringsAsFactors = FALSE)

head(SuperstoreSales_main)
head(SuperstoreSales_manager)
head(SuperstoreSales_returns)

# make Product table ’product’:
product <- SuperstoreSales_main %>%
  select(`Product.Name`, `Product.Sub.Category`, `Product.Category`) %>%
  rename(name = `Product.Name`, sub_category=`Product.Sub.Category`,category=`Product.Category`) %>%
  arrange(name, sub_category) %>%
  group_by(name, sub_category) %>%
  distinct() %>%
  ungroup() %>%
  mutate(product_id = row_number())

# make Customer table ’customer’:
customer <- SuperstoreSales_main %>%
  select(`Customer.Name`, `Customer.Segment`, `Province`, `Region`) %>%
  rename(name = `Customer.Name`, segment = `Customer.Segment`, province = `Province`, region=`Region`) %>%
  arrange(name, segment) %>%
  group_by(name, segment) %>%
  distinct() %>%
  ungroup() %>%
  mutate(cust_id = row_number())

# make Sales table ’sales’:
sales <- SuperstoreSales_main %>%
  select(`Order.Date`, `Sales`, `Order.Quantity`, `Unit.Price`, `Profit`, `Shipping.Cost`, `Ship.Date`, `Order.ID`, `Product.Name`, `Product.Sub.Category`, `Product.Category`, `Customer.Name`, `Customer.Segment`, `Province`, `Region`) %>%
  rename(sales=`Sales`, order_date=`Order.Date`, order_qty=`Order.Quantity`, unitprice=`Unit.Price`, ship_date=`Ship.Date`, profit=`Profit`, shipping_cost = `Shipping.Cost`)

sales <- sales %>%
  full_join(product, by = c("Product.Name" = "name",
                            "Product.Sub.Category" = "sub_category",
                            "Product.Category" = "category")) %>%
  select( -`Product.Name`, -`Product.Sub.Category`, -`Product.Category`)

sales <- sales %>%
  full_join(customer, by = c("Customer.Name" = "name",
                             "Customer.Segment" = "segment",
                             "Province" = "province",
                             "Region" = "region")) %>%
  select( -`Customer.Name`, -`Customer.Segment`, -`Province`, -`Region`)

sales <- sales %>%
  full_join(SuperstoreSales_returns, by = c("Order.ID" = "Order.ID")) %>%
  select( -`Order.ID`)

sales$order_date = as.Date(as.character(sales$order_date), "%d/%m/%y",tz="UTC")
sales$ship_date = as.Date(as.character(sales$ship_date), "%d/%m/%y",tz="UTC")
sales$shipped_late<-ifelse(difftime(sales$ship_date,sales$order_date,units = "days") >2, 1, 0)

sales$Status[is.na(sales$Status)] <- 0
sales$returned<-ifelse(sales$Status == "Returned", 1, 0)

sales <- sales %>% 
  select(-`Status`, -`ship_date`)

sales_sums <- sales %>%
  group_by(`product_id`, `order_date`, `shipped_late`, `returned`, `cust_id`) %>%
  summarise_at(vars("order_qty","sales", "profit", "shipping_cost"), sum)

sales2 <- sales %>%
  select(`product_id`, `order_date`, `shipped_late`, `returned`, `cust_id`, `unitprice`) %>%
  group_by(`product_id`, `order_date`, `shipped_late`, `returned`, `cust_id`) %>%
  distinct() %>%
  ungroup() %>%
  full_join(sales_sums, by=c("product_id", "order_date", "shipped_late", "returned", "cust_id"))
  
  # summarise(sales = sum(sales), orderquantity = sum(orderquantity), unitprice=mean(unitprice), profit = sum(profit), shippingcost = sum(shippingcost))

# least profitable product
leastprofprod <- sales %>% 
  select(profit, product_id) %>%
  aggregate(by = list(sales$product_id), FUN = sum) %>%
  select(Group.1, profit) %>%
  rename(product_id = Group.1) %>%
  arrange(profit) %>%
  full_join(product, by = c("product_id" = "product_id")) %>% 
  select(name, profit)



# plot of profit vs product
ggplot(data=head(leastprofprod, 5), aes(x = reorder(`name`, `profit`), y = `profit`)) + geom_bar(stat = "identity") + xlab("product") + scale_x_discrete(labels = function(x) str_wrap(x, width = 10))

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

