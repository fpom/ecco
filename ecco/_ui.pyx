import time, datetime
import psutil
import ipywidgets as ipw
from IPython.display import display

cdef class Logger (object) :
    cdef public bint verbose
    cdef public str head, tail, done_head, done_tail
    cdef unsigned int total, done, count, step, start
    cdef bint keep
    cdef float progress
    cdef list steps
    cdef object hbox, whead, bar, wtail
    def __init__ (self, *l, **k) :
        self(*l, **k)
    def __call__ (self, head="", tail="", done_head="", done_tail="",
                  total=0, steps=[], keep=False, verbose=None) :
        if verbose is not None :
            self.verbose = verbose
        if not self.verbose :
            return self
        self.head = head
        self.tail = tail
        self.done_head = done_head
        self.done_tail = done_tail
        self.total = total
        self.steps = steps
        self.keep = keep
        if self.steps :
            self.total = len(steps)
        self.start = int(time.time())
        self.progress = 0
        self.count = self.done = 0
        self.step = max(1, self.total // 100)
        self.whead = ipw.HTML(value="")
        self.wtail = ipw.HTML(value="")
        if self.total :
            self.bar = ipw.IntProgress(min=0, max=self.total, value=0, bar_style="info")
            self.hbox = ipw.HBox([self.whead, self.bar, self.wtail])
        else :
            self.hbox = ipw.HBox([self.whead, self.wtail])
        return self
    cdef inline str _fmt (Logger self, str s) :
        cdef object mem = psutil.virtual_memory()
        cdef str current_step
        cdef int elapsed
        if self.steps and 0 <= self.done < len(self.steps) :
            current_step = self.steps[self.done]
        elif self.steps and self.done == len(self.steps) :
            current_step = "(all done)"
        else :
            current_step = "(working)"
        elapsed = int(time.time()) - self.start
        if self.total and 1 < self.done <= self.total :
            eta = datetime.timedelta(seconds=(elapsed * (self.total - self.done)
                                              // self.done))
        else :
            eta = "???"
        return s.format(total=self.total,
                        done=self.done,
                        step=current_step,
                        progress=self.progress,
                        time=datetime.timedelta(seconds=elapsed),
                        eta=eta,
                        memory=100.0 * (1.0 - mem.available / mem.total))
    cdef inline void _update (self) :
        if self.total :
            self.bar.value = self.done
            if self.done > self.total :
                self.bar.bar_style = "warning"
            self.progress = 100 * (self.done / self.total)
        self.whead.value = self._fmt(self.head)
        self.wtail.value = self._fmt(self.tail)
    def __enter__ (self) :
        if not self.verbose :
            return self
        self.start = int(time.time())
        self._update()
        display(self.hbox)
        return self
    def __exit__ (self, exc_type, exc_val, exc_tb) :
        if not self.verbose :
            return
        if self.total and exc_type is not None :
            self.bar.bar_style = "danger"
        elif self.total and self.done != self.total :
            self.bar.bar_style = "warning"
        elif self.total :
            self.bar.close()
        self.whead.value = self._fmt(self.done_head or self.head)
        self.wtail.value = self._fmt(self.done_tail or self.tail)
        if not self.keep :
            self.hbox.close()
    cpdef void update (Logger self, unsigned int count=1) :
        if not self.verbose :
            return
        self.update_to(self.done + self.count + count)
    cpdef void update_to (Logger self, unsigned int done) :
        if not self.verbose :
            return
        if self.done > done :
            # go backward
            self.done = done
            self.count = 0
            self._update()
        else :
            self.count += done - self.done
            if self.count >= self.step :
                # enough accumulated small updates
                self.done += self.count
                self.count = 0
                self._update()
    def finish (Logger self, **update) :
        self.done = self.total
        for name, value in update.items() :
            setattr(self, name, value)
    cpdef void print (Logger self, str message, str level="") :
        cdef dict style = {"[info]" : {"color" : "#008800"},
                           "[warning]" : {"color" : "#BF7C00"},
                           "[error]" : {"color" : "#008800"}}
        cdef dict default = {"color" : "#000088"}
        if level :
            display(ipw.HTML('<p style="line-height:140%">'
                             '<span style="color:{color}; font-weight:bold;">'
                             '{level}</span> {message}'
                             '</p>'.format(message=message,
                                           level=level,
                                           **style.get(level, default))))
        else :
            display(ipw.HTML('<p style="line-height:140%">{message}'
                             '</p>'.format(message=message)))
    cpdef void info (Logger self, str message) :
        self.print(message, "[info]")
    cpdef void warn (Logger self, str message) :
        self.print(message, "[warning]")
    cpdef void err (Logger self, str message) :
        self.print(message, "[error]")
