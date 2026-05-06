Before submitting a PR the following must be true:
1. There's a single script a user can run to make a submission, based on an implemented model interface.
2. Ensure the total submission time fits in under 10mins
    (with 5mins training and ~3mins for provisioning)


# Open questions:

## How to deal with GPU availabilty?

a) keep a GPU running while generating experiments

b) generate multiple experiments, batch train/evaluate when GPU becomes available.
