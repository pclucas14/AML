for (new_x, new_y) in stream:
    old_x, old_y = buffer.sample()

    x = concat((new_x, old_x))
    y = concat((new_y, old_y))

    loss = loss_fn(model(x), y)

    optimizer(model, loss)

    buffer.add(new_x, new_y)

for (new_x, new_y) in stream:
    old_x, old_y = buffer.sample()

    new_loss = new_loss_fn(model(new_x), new_y)
    old_loss = old_loss_fn(model(old_x), old_y)

    loss = (new_loss + old_loss) / 2

    optimizer(model, loss)

    buffer.add(new_x, new_y)
