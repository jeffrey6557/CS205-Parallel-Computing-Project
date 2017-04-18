# Cuda
cuda_loss = c(1.3733631402355129, 0.93689111362069344, 0.91562216390997675, 0.90943513039815227, 0.90553493915978123, 0.90839237429327879, 0.89844868593296767, 0.89804859597804187, 0.89566163630808815, 0.89294650195008618, 0.89129256531343626, 0.89407325882022659, 0.89177121463468523, 0.88653495932029469, 0.88972305944410424, 0.89187817670531189, 0.88975714647163784, 0.89168526544409288, 0.89036472449868409, 0.88598235974877571)
cuda_valLoss = c(0.80806828995390856, 0.79556096959461486, 0.78946638959660365, 0.78453055065553179, 0.78353461688835713, 0.78164274223535113, 0.78354178043350609, 0.78122799726148173, 0.78135497052562075, 0.78002393609263132, 0.77844687327818318, 0.77849215448851994, 0.77751679577612542, 0.77562201441202838, 0.77743307886104263, 0.77637484535367884, 0.77685708831585454, 0.77796572454902368, 0.77658879340846265, 0.77585375844240589)

# OpenMP
mp_loss = c(2.0860142872697214, 0.9895793886507972, 0.96470873422541858, 0.94888066522145675, 0.94124168347504178, 0.94033507177385234, 0.93775131500373454, 0.93218556565753485, 0.93068114104917499, 0.92750692610013286, 0.93028744879415481, 0.92693335561429036, 0.92466088274777947, 0.92795208322799816)

mp_valLoss = c(0.84147622463299998, 0.79811348312388275, 0.79394501364865333, 0.79243479143438078, 0.79133199328529513, 0.79077467066843365, 0.79094408601471955, 0.79070669729723364, 0.79071738137426473, 0.79098788910661866, 0.79085519927027104, 0.79129322889658282, 0.79139959153705675, 0.79173009575766573)

# Sequential
seq_loss = c(1.076783147585594, 1.058036088620202, 1.0555443448737516, 1.0551168598158884, 1.05667500432265, 1.050544273247153, 1.0543573314052517, 1.0553697162563518, 1.0516507674071749, 1.0518347119476836, 1.0517346988290044, 1.0514595026080891, 1.0525540688886481, 1.0507191341206179, 1.0507543346841457, 1.0495227093817825, 1.0493365861399699, 1.0506070852683762, 1.0480825689485518, 1.0496527643122915, 1.0487889635764946, 1.047232785548194, 1.0478389479952344, 1.0491301786293419, 1.046774112733744, 1.046920451592591)
seq_valLoss = c(0.94508946055034504, 0.94031213914204836, 0.93605821026013847, 0.9366118094752256, 0.93616342398119312, 0.93389519276517163, 0.93615947655199827, 0.93545188273829305, 0.93744580043206494, 0.9363785574230975, 0.93378809116646944, 0.93765381367398049, 0.93397481481490074, 0.93297016766062435, 0.93262579620025876, 0.9319136054363949, 0.93119008756823074, 0.93121627033746512, 0.93353505599745168, 0.92992675672018577, 0.93122863801848244, 0.93088368731650284, 0.93155695917605541, 0.93165850122271454, 0.93110575099510084, 0.9335181804904692)

epochs = 1:26
cuda_loss = c(cuda_loss, rep(NA,6)); cuda_valLoss = c(cuda_valLoss, rep(NA,6))
mp_loss = c(mp_loss, rep(NA,12)); mp_valLoss = c(mp_valLoss, rep(NA,12))

### Plot training loss
pdf("trainingLoss_plot.pdf", width = 3.5, height = 3, pointsize = .7)
plot(epochs, seq_loss, type = "l", lwd = 2, ylim = c(min(cuda_loss, seq_loss, mp_loss, na.rm = T), 
                                                     max(cuda_loss, seq_loss, mp_loss, na.rm = T)),
     xlab = "Number of Epoch", ylab = "Training Loss")
lines(epochs, cuda_loss, col = 2, lwd = 2)
lines(epochs, mp_loss, col = 3, lwd = 2)
legend("topright", legend = c("Sequential","Cuda","OpenMP"), col = 1:3, lwd = 2)
dev.off()

### Plot validation loss
pdf("validationLoss_plot.pdf", width = 3.5, height = 3, pointsize = .7)
plot(epochs, seq_valLoss, type = "l", lwd = 2, ylim = c(min(cuda_valLoss, seq_valLoss, mp_valLoss, na.rm = T), 
                                                     max(cuda_valLoss, seq_valLoss, mp_valLoss, na.rm = T)),
     xlab = "Number of Epoch", ylab = "Validation Loss")
lines(epochs, cuda_valLoss, col = 2, lwd = 2)
lines(epochs, mp_valLoss, col = 3, lwd = 2)
legend("right", legend = c("Sequential","Cuda","OpenMP"), col = 1:3, lwd = 2)
dev.off()

### Plot confidence intervals
xlabs = c("Accuracy", "Hit Ratio", "Mean squared error","Mean absolute error")
model1Frame <- data.frame(Metrics = xlabs,
                          Estimate = c(0.669196827979, 0.620052877771, 0.521886447239, 0.429926624148),
                          SE = c(0.1729662, 0.3093834, 0.5672855, 0.178873),
                          Algorithm = "Sequential algorithm")
model2Frame <- data.frame(Metrics = xlabs,
                          Estimate = c(0.71378405856, 0.702302216799, 0.330157280831, 0.34997900652),
                          SE = c(0.1878927,0.2737856,0.4100375,0.1977138),
                          Algorithm = "OpenMP")
model3Frame <- data.frame(Metrics = xlabs,
                          Estimate = c(.701785278569, .695375228798, .275805540997, .32731569056),
                          SE = c(0.1574084, 0.2474334, 0.3783008, 0.1866145),
                          Algorithm = "CUDA")
# Combine these data.frames
allModelFrame <- data.frame(rbind(model1Frame, model2Frame, model3Frame))  # etc.

# Specify the width of your confidence intervals
interval1 <- -qnorm((1-0.9)/2)  # 90% multiplier
interval2 <- -qnorm((1-0.95)/2)  # 95% multiplier

# Plot
pdf("error_metrics.pdf", width = 6, height = 6, pointsize = .7)
zp1 <- ggplot(allModelFrame, aes(colour = Algorithm))
zp1 <- zp1 + geom_hline(yintercept = 0, colour = gray(1/2), lty = 2)
zp1 <- zp1 + geom_linerange(aes(x = Metrics, ymin = Estimate - SE*interval1/interval2,
                                ymax = Estimate + SE*interval1/interval2),
                            lwd = 1, position = position_dodge(width = 1/2))
zp1 <- zp1 + geom_pointrange(aes(x = Metrics, y = Estimate, ymin = Estimate - SE,
                                 ymax = Estimate + SE),
                             lwd = 1/2, position = position_dodge(width = 1/2),
                             shape = 21, fill = "WHITE")
zp1 <- zp1 + coord_flip() + theme_bw()
zp1 <- zp1 + ggtitle("Comparison of Algorithms")
print(zp1)
dev.off()
