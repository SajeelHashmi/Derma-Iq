from django.shortcuts import render , redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login
from django.contrib.auth import logout
from django.core.validators import EmailValidator
from django.conf import settings
from django.contrib import messages

"""Basic auth complete add stripe subscription to register view and check subscription on login view"""

def login_view(request):
    if request.method == 'POST':
        if not request.user.is_authenticated:
            username = request.POST.get('username')
            password = request.POST.get('pass')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("home")
            else:
                messages.success(request=request,message="incorrect user name or password")
                return redirect('login')
        else:
            return render(request , 'login.html', {})
    return render( request, 'login.html' )

def logout_user(request):
    logout(request=request)
    return redirect("login")

    
def sign_up(request):
    if request.user.is_authenticated:
        return redirect("home")
    if request.method =="POST":

        # print("method post")
        username = request.POST["username"]
        email = request.POST["email"]
        pass1 = request.POST["passowrd"]
        pass2 = request.POST["confirm_password"]

        normalized_email = email.lower()   
        # validate the data in the form
        if pass1==pass2:
            user = User.objects.create_user(username=username, password=pass1,email=normalized_email)
            user.save()
            login(request,user)
            return redirect("home")
        else:
            messages.success(request=request,message="PASSWORDS DONOT MATCH")
    return render(request,"signup.html")



def home(request):
    if request.user.is_authenticated:
        # This will show the users previous scans and other data
        # A basic dashboard with new scan button and below that the previous scans
        # also somewhere in between tips and tricks for better face health
        
        return render(request, 'home.html', {})
    else:
        return redirect('login')