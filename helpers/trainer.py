class Callback:
    """Base class for creating custom callbacks to hook into the training process."""

    def on_train_start(self, history):
        """Called once at the beginning of training."""
        pass

    def on_epoch_start(self, epoch):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.
        Logs typically contains the training/validation loss and metrics.
        """
        pass

    def on_train_end(self, history):
        """Called once at the end of training."""
        pass


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader=None,
        device=None,
        log_dir=None,
        callbacks=None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        self._log_dir = log_dir
        self._callbacks = callbacks if callbacks is not None else []

    def _train_step(self, epoch, epochs, writer):
        """Training phase in an epoch."""
        self._model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(self._train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(self._device), labels.to(self._device)

            # Forward pass
            outputs = self._model(inputs)
            loss = self._criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()  # compute the gradient of the loss
            optimizer.step()  # update the model parameters

            # Logging batch-level metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update tqdm progress bar description
            train_pbar.set_postfix({"loss": loss.item()})

            # Log batch loss to TensorBoard
            global_step = epoch * len(train_loader) + i
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", train_accuracy, epoch)

    def _validation_step(self, epoch, epochs, writer):
        """Validation phase in an epoch."""
        self._model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(self._val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = self._criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({"loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

        writer.add_scalar("Loss/validation_epoch", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/validation_epoch", val_accuracy, epoch)

    def train(self, epochs):
        """Main method to start the model training."""
        print(f"Starting training on '{self._device}'...")
        log_dir = self._log_dir
        if log_dir is None:
            log_dir = f"logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}"
        writer = SummaryWriter(log_dir)
        print(f"Logs will be saved to {log_dir}")

        for callback in self._callbacks:
            callback.on_train_start(history)

        train_start_t = time.time()
        for epoch in range(epochs):
            for callback in self._callbacks:
                callback.on_epoch_start(epoch)

            # --- Training Phase ---
            epoch_start_t = time.time()
            self._train_step(epoch, epochs, writer)
            train_epoch_stop_t = time.time()

            # --- Validation Phase ---
            if self._val_dataloader:
                self._validation_step(epoch, epochs, writer)
            epoch_end_t = time.time()

            # --- End-of-Epoch Logging ---
            train_duration = train_epoch_stop_t - epoch_start_t
            val_duration = epoch_end_t - train_epoch_stop_t
            epoch_duration = epoch_end_t - epoch_start_t
            val_log = f"Validation Time: {val_duration:.2f}s | " if self._val_dataloader else ""
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Time: {train_duration:.2f}s | {val_log}"
                f"Epoch Time: {epoch_duration:.2f}s"
            )
            writer.add_scalar("Learning_Rate", self._optimizer.param_groups[0]["lr"], epoch)

            for callback in self._callbacks:
                callback.on_epoch_end(epoch, logs)

        total_train_duration = time.time() - train_start_t
        print(
            f"Total Training Time: {str(datetime.timedelta(seconds=total_train_duration))} | {total_train_duration:.2f}s"
        )
        for callback in self._callbacks:
            callback.on_train_end(history)

        print("Training complete.")

        writer.close()