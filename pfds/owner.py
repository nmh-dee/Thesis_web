from django.views.generic import CreateView, UpdateView, ListView, DeleteView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin

class OwnerListView(ListView):
    """
    Sub-class the ListView to pass the request to the form.
    """
class OwnerDetailView(DetailView):
    """
    Sub-class the DetailView to pass the request to the form.
    """


class OwnerUpdateView(LoginRequiredMixin,UpdateView):
    def get_queryset(self):
        print('update get_queryset called')
        """ Limit a User to only modifying their own data. """
        qs = super(OwnerUpdateView, self).get_queryset()
        return qs.filter(owner=self.request.user)
class OwnerDeleteView(LoginRequiredMixin, DeleteView):
    """
    Sub-class the DeleteView to restrict a User from deleting other
    user's data.
    """
    def get_queryset(self):
        print('delete get_queryset called')
        qs = super(OwnerDeleteView, self).get_queryset()
        return qs.filter(owner=self.request.user) 