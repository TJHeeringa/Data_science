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

sales <- sales %>%
  select(`product_id`, `order_date`, `shipped_late`, `returned`, `cust_id`, `unitprice`) %>%
  group_by(`product_id`, `order_date`, `shipped_late`, `returned`, `cust_id`) %>%
  distinct() %>%
  ungroup() %>%
  full_join(sales_sums, by=c("product_id", "order_date", "shipped_late", "returned", "cust_id"))
  
  # summarise(sales = sum(sales), orderquantity = sum(orderquantity), unitprice=mean(unitprice), profit = sum(profit), shippingcost = sum(shippingcost))

# least profitable product sub-category
leastprofprod <- sales %>%
  full_join(product, by = c("product_id" = "product_id")) %>% 
  select(name, profit) %>%
  group_by(name) %>%
  summarise(profit = sum(profit)) %>%
  arrange(profit)

# plot of profit vs product
lppplot <- ggplot(data=head(leastprofprod, 5), aes(x = reorder(`name`, `profit`), y = `profit`)) + geom_bar(stat = "identity") + xlab("product") + scale_x_discrete(labels = function(x) str_wrap(x, width = 20))
ggsave(filename = "Ass3 - 1 - Least Profitable Product.pdf", plot = lppplot)

# least profitable product sub-category
leastprofcat <- sales %>%
  full_join(product, by = c("product_id" = "product_id")) %>% 
  select(sub_category, profit) %>%
  group_by(sub_category) %>%
  summarise(profit = sum(profit)) %>%
  arrange(profit)

lpcplot <- ggplot(data=head(leastprofcat, 5), aes(x = reorder(`sub_category`, `profit`), y = `profit`)) + geom_bar(stat = "identity") + xlab("sub-category") + scale_x_discrete(labels = function(x) str_wrap(x, width = 20))
ggsave(filename = "Ass3 - 2 - Least Profitable Category.pdf", plot = lpcplot)


lateprod <- sales %>%
  full_join(product, by = c("product_id" = "product_id")) %>% 
  select(name, shipped_late) %>%
  group_by(name) %>%
  summarise(shipped_late = sum(shipped_late)) %>%
  arrange(-shipped_late)

laprplot <- ggplot(data=head(lateprod, 5), aes(x = reorder(`name`, -`shipped_late`), y = `shipped_late`)) + geom_bar(stat = "identity") + xlab("Product") + ylab("shipped late (#)") + ggtitle("Products that were shipped late the most often") + scale_x_discrete(labels = function(x) str_wrap(x, width = 20))
ggsave(filename = "Ass3 - 3 - Late Products.pdf", plot = laprplot)

latecat <- sales %>%
  full_join(product, by = c("product_id" = "product_id")) %>% 
  select(sub_category, shipped_late) %>%
  group_by(sub_category) %>%
  summarise(shipped_late = mean(shipped_late)*100) %>%
  arrange(-shipped_late)

lacaplot <- ggplot(data=head(latecat, 5), aes(x = reorder(`sub_category`, -`shipped_late`), y = `shipped_late`)) + geom_bar(stat = "identity") + xlab("Sub-Category") + ylab("shipped late (%)") + ggtitle("Sub-categories that were shipped late the most often") + scale_x_discrete(labels = function(x) str_wrap(x, width = 20))
ggsave(filename = "Ass3 - 4 - Late Categories.pdf", plot = lacaplot)

returnprod <- sales %>%
  full_join(product, by = c("product_id" = "product_id")) %>% 
  select(name, returned) %>%
  group_by(name) %>%
  summarise(returned = sum(returned)) %>%
  arrange(-returned)

reprplot <- ggplot(data=head(returnprod, 6), aes(x = reorder(`name`, -`returned`), y = `returned`)) + geom_bar(stat = "identity") + xlab("Product") + ylab("Returned (#)") + ggtitle("Products that were returned the most often") + scale_x_discrete(labels = function(x) str_wrap(x, width = 20))
ggsave(filename = "Ass3 - 5 - Returned Products.pdf", plot = reprplot)

returncat <- sales %>%
  full_join(product, by = c("product_id" = "product_id")) %>% 
  select(sub_category, returned) %>%
  group_by(sub_category) %>%
  summarise(returned = mean(returned)*100) %>%
  arrange(-returned)

recaplot <- ggplot(data=head(returncat, 5), aes(x = reorder(`sub_category`, -`returned`), y = `returned`)) + geom_bar(stat = "identity") + xlab("Sub-category") + ylab("returned (%)") + ggtitle("Sub-Categories that were returned the most often") + scale_x_discrete(labels = function(x) str_wrap(x, width = 20))
ggsave(filename = "Ass3 - 6 - Returned Categories.pdf", plot = recaplot)


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
dbGetQuery(con,
           "SELECT table_name FROM information_schema.tables
            WHERE table_schema=’ass3’")
