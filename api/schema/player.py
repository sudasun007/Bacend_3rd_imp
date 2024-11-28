from marshmallow import Schema, fields

class PlayerSchema(Schema):
    Email = fields.Email(required=True)
    F_name = fields.Str(required=True)
    L_name = fields.Str(required=True)
    Password = fields.Str(required=True, load_only=True)
    Active_status = fields.Bool(dump_only=True)

class LoginSchema(Schema):
    Email = fields.Email(required=True)
    Password = fields.Str(required=True, load_only=True)
