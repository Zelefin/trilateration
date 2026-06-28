CC ?= gcc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O2
CPPFLAGS := -I../components/trilat/include
LDLIBS := -lm

TEST_BIN := test_trilat
SRCS := test_trilat.c ../components/trilat/trilat.c

.PHONY: test clean

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(SRCS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(SRCS) $(LDLIBS)

clean:
	rm -f $(TEST_BIN)
