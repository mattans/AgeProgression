import model
import consts

if 'net' not in globals():  # for interactive execution in PyCharm
    net = model.Net()
    net.to(device=consts.device)

    print(consts.device)

    MOCK_TEST = True
    if MOCK_TEST:
        net.train(consts.IMAGE_PATH, batch_size=consts.BATCH_SIZE, epochs=consts.EPOCHS)
        net.save("./trained_models")
