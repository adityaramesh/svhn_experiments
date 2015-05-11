ADADELTA_RHO_LIST=0.9 0.95 0.99 0.995
ADADELTA_EPS_LIST=1e-2 1e-4 1e-6 1e-8 1e-10
SGD_EPS=1e0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6
DEVICE=2
N_EPOCH=2


adadelta_rho:
	for rho in ${ADADELTA_RHO_LIST}; do \
		th source/drivers/svhn_5x5_baseline.lua -task replace -model 5x5_adadelta_rho$${rho} -device ${DEVICE} -n_epoch ${N_EPOCH} -adadelta_eps 1e-10 -adadelta_rho $${rho}; \
		rm models/5x5_adadelta_rho$${rho}/*.t7; \
	done

adadelta_eps:
	for eps in ${ADADELTA_EPS_LIST}; do \
                th source/drivers/svhn_5x5_baseline.lua -task replace -model 5x5_adadelta_eps$${eps} -device ${DEVICE} -n_epoch ${N_EPOCH} -adadelta_eps $${eps} -adadelta_rho 0.99; \
                rm models/5x5_adadelta_eps$${eps}/*.t7; \
        done

plain_sgd:
	for eps in ${SGD_EPS}; do \
		th source/drivers/svhn_5x5_baseline.lua -task replace -model 5x5_adadelta_eps$${eps} -device ${DEVICE} -n_epoch ${N_EPOCH} -sgd_eps $${eps}; \ 
		rm models/5x5_adadelta_eps$${eps}/*.t7; \
	done

test:
	th -l cunn

