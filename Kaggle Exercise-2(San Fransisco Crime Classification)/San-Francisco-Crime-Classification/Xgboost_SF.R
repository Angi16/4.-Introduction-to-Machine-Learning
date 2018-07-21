#Reading the training data
train = fread("train.csv")
test = fread("test.csv")

#Loading the required Libraries
library(data.table)
library(xgboost)
library(caret)

#Merge the train & test data
data = merge(train, test, by=c("Dates", "DayOfWeek", "Address","X","Y","PdDistrict"), all=TRUE)
dim(data)
a = dim(data)[1]

#Dropping columns descript and resolution
columns <- c('Descript', 'Resolution')
for (column in columns){
  data[[column]] <- NULL
}

#Splitting the datetime field
datetime <- strptime(data$Dates, format="%m/%d/%Y %H:%M")
data$Year <- as.numeric(format(datetime, "%Y"))
data$Month <- as.numeric(format(datetime, "%m"))
data$Day <- as.numeric(format(datetime, "%d"))
data$Hour <- as.numeric(format(datetime, "%H"))
data$Minute <- as.numeric(format(datetime, "%M"))

#Address
data$intersection = ifelse((grepl("Block", data$Address)),1,0)

#PCA
Sep <- with(data, which(Y == 90))
transform <- preProcess(data[-Sep, c('X', 'Y'), with=FALSE], method = c('center', 'scale', 'pca'))
pc <- predict(transform, data[, c('X', 'Y'), with=FALSE]) 
data$X <- pc$PC1
data$Y <- pc$PC2

#Morning,Afternoon,Evening and Dark Features
data$Morning = ifelse((data$Hour >=6 && data$Hour <12),1,0)
data$Afternoon = ifelse((data$Hour >=12 && data$Hour <17),1,0)
data$Evening = ifelse((data$Hour >=17 && data$Hour <20),1,0)
data$Dark = ifelse((data$Hour >=20 && data$Hour <6),1,0)

#Fall,Winter,Spring,Summer
data$Fall = ifelse((data$Month >=3 && data$Month<=5),1,0)
data$Winter = ifelse((data$Month >=6 && data$Month<=8),1,0)
data$Spring = ifelse((data$Month >=9 && data$Month<=11),1,0)
data$Summer = ifelse((data$Month >=12 && data$Month<=2),1,0)

# test/train separation
Sep <- which(!is.na(data$Category))
classes <- sort(unique(data[Sep]$Category))
m <- length(classes)
data$Class <- as.integer(factor(data$Category, levels=classes)) - 1
dim(data)

#Converting non-numeric features to numeric
feature.names <- names(data)[which(!(names(data) %in% c('Id', 'Address', 'Dates', 'Category', 'Class')))]
for (feature in feature.names){
  if (class(data[[feature]]) == 'character'){
    cat(feature, 'converted\n')
    levels <- unique(data[[feature]])
    data[[feature]] <- as.integer(factor(data[[feature]], levels=levels))
  }
}

#Params for XGboost
param <- list(
  #nthread             = 4,
  booster             = 'gbtree',
  objective           = 'multi:softprob',
  num_class           = m,
  eta                 = 1,
  #gamma               = 0,
  max_depth           = 6,
  #min_child_weigth    = 1,
  max_delta_step      = 1
  #subsample           = 1,
  #colsample_bytree    = 1,
  #early.stop.round    = 5
)
ext <- sample(1:length(Sep), floor(9*length(Sep)/10))
dval <- xgb.DMatrix(data=data.matrix(data[Sep[-ext], feature.names, with=FALSE]), label=data[Sep[-ext]]$Class)
dtrain <- xgb.DMatrix(data=data.matrix(data[Sep[ext], feature.names, with=FALSE]), label=data[Sep[ext]]$Class)
watchlist <- list(val=dval, train=dtrain)
Xgbst <- xgb.train( params            = param,
                  data              = dtrain,
                  watchlist         = watchlist,
                  verbose           = 1,
                  eval_metric       = 'mlogloss',
                  nrounds           = 18
)
dtest <- xgb.DMatrix(data=data.matrix(data[-Sep,][order(Id)][,feature.names, with=FALSE]))
prediction <- predict(Xgbst, dtest)
prediction <- sprintf('%f', prediction)
prediction <- cbind(data[-Sep][order(Id)]$Id, t(matrix(prediction, nrow=m)))
dim(prediction)

colnames(prediction) <- c('Id', classes)
#names(prediction)
write.csv(prediction, 'submission.csv', row.names=FALSE, quote=FALSE)

