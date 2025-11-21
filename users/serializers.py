from .models import User
from rest_framework import serializers


class RegisterSerializer(serializers.ModelSerializer):
    confirm_password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['name', 'lastName', 'email', 'password', 'confirm_password']
        extra_kwargs = {'password': {'write_only': True}}

    def validate(self, data):
        if data['password'] != data['confirm_password']:
            raise serializers.ValidationError({"confirm_password": "Passwords must match."})
        return data

    def create(self, validated_data):
        validated_data.pop('confirm_password')
        user = User.objects.create_user(
            email=validated_data['email'],
            name=validated_data['name'],
            last_name=validated_data['lastName'],
            password=validated_data['password']
        )
        return user


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['name', 'lastName', 'email']
        read_only_fields = ['email']

    def validate_name(self, value):
        if not value.isalpha():
            raise serializers.ValidationError("First name should contain only letters.")
        return value.title()

    def validate_lastName(self, value):
        if not value.replace('-', '').isalpha():
            raise serializers.ValidationError("Last name should contain only letters.")
        return value.title()
