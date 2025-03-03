#install.packages(c("keras3", "tensorflow", "magick", "gridExtra"))

library(keras3)
library(tensorflow)
library(magick)
library(gridExtra)
library(grid)

rm(list=ls())

# Check for GPU and configure TensorFlow to use it
physical_devices <- tf$config$list_physical_devices("GPU")
if (length(physical_devices) > 0) {
  tf$config$experimental$set_memory_growth(physical_devices[[1]], TRUE)
  cat("GPU found and configured.\n")
} else {
  cat("No GPU found, using CPU.\n")
}

# Load CIFAR-10 data
cifar10 <- dataset_cifar10()
x_train <- cifar10$train$x
y_train <- cifar10$train$y
x_test <- cifar10$test$x
y_test <- cifar10$test$y

# Normalize pixel values to the range [0, 1]
x_train <- x_train / 255
x_test <- x_test / 255

# Convert class vectors to binary class matrices (one-hot encoding)
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Define the model (Convolutional Neural Network)
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(32, 32, 3)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")

# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)

# Define the TensorBoard callback
tensorboard_callback <- callback_tensorboard(
  log_dir = "logs/cifar10_logs",
  histogram_freq = 1
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 15,
  batch_size = 128,
  validation_split = 0.2,
  callbacks = list(tensorboard_callback)
)

# Evaluate the model
score <- model %>% evaluate(x_test, y_test)
cat("Test loss:", score$loss, "\n")
cat("Test accuracy:", score$accuracy, "\n")



#Demonstrate model predictions
num_images_to_display <- 9
indices <- sample(1:nrow(x_test), num_images_to_display)
selected_images <- x_test[indices, , , ]
predictions <- model %>% predict(selected_images)

#some images have label probability as 0
#class label conversion
predicted_classes <- apply(predictions, 1, function(row) {
  if (all(row == 0)) {
    return(NA) # Handle cases where all probabilities are zero
  } else {
    return(which.max(row) - 1)
  }
})

actual_classes <- apply(y_test[indices, ], 1, function(row) {
  if (all(row == 0)) {
    return(NA) # Handle cases where all probabilities are zero
  } else {
    return(which.max(row) - 1)
  }
})

class_names <- c("airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck")

image_list <- list()

for (i in 1:num_images_to_display) {
  img <- as.raster(selected_images[i, , , ])
  p <- rasterGrob(img)
  predicted_name <- if (!is.na(predicted_classes[i])) class_names[predicted_classes[i] + 1] else "NA"
  actual_name <- if (!is.na(actual_classes[i])) class_names[actual_classes[i] + 1] else "NA"
  
  title <- paste0("Predicted: ", predicted_name,
                  "\nActual: ", actual_name)
  image_list[[i]] <- arrangeGrob(p, top = textGrob(title, gp = gpar(fontsize=10)))
}
#Arrange in a grid
grid.arrange(grobs = image_list, ncol = floor(sqrt(num_images_to_display)))
